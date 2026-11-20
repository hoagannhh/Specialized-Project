"""Chạy giao diện local: python demo_app/app.py --port 8501."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from demo_app.service import capabilities, analyze, OUTPUT

STATIC=Path(__file__).parent/'static'
JOBS={};LOCK=threading.Lock();POOL=ThreadPoolExecutor(max_workers=1)
TOKEN=secrets.token_urlsafe(32)


def worker(job_id,payload):
    def progress(message):
        with LOCK:JOBS[job_id]['message']=message
    try:
        result=analyze(payload,job_id,progress)
        with LOCK:JOBS[job_id].update(status='done',message='Hoàn tất',result=result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        message=str(exc) if isinstance(exc,(ValueError,FileNotFoundError)) else 'Không thể hoàn tất phân tích. Xem terminal để biết chi tiết; kiểm tra checkpoint và kết nối dịch vụ.'
        with LOCK:JOBS[job_id].update(status='error',message=message)


class Handler(BaseHTTPRequestHandler):
    def send(self, data, mime='application/json; charset=utf-8', status=200, attachment=None):
        if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode('utf8')
        elif isinstance(data,str):data=data.encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        if attachment:self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
        self.end_headers();self.wfile.write(data)

    def allowed_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.allowed_host():return self.send({'error':'Local access only'},status=403)
        path=urlparse(self.path).path
        files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
        if path in files:
            name,mime=files[path];return self.send((STATIC/name).read_bytes(),mime)
        if path=='/api/status':return self.send({**capabilities(),'token':TOKEN})
        parts=path.strip('/').split('/')
        if len(parts) in (3,4) and parts[:2]==['api','jobs']:
            with LOCK:job=dict(JOBS.get(parts[2],{}))
            if not job:return self.send({'error':'Không tìm thấy phiên phân tích'},status=404)
            if len(parts)==3:return self.send(job)
            if parts[3] in ('report.json','report.html','report.docx') and job['status']=='done':
                file=OUTPUT/parts[2]/parts[3]
                if file.is_file():return self.send(file.read_bytes(),'application/octet-stream',attachment=parts[3])
        self.send({'error':'Không tìm thấy'},status=404)

    def do_POST(self):
        if not self.allowed_host() or self.headers.get('X-Demo-Token')!=TOKEN:
            return self.send({'error':'Hãy tải lại trang để mở phiên local'},status=403)
        if urlparse(self.path).path!='/api/analyze':return self.send({'error':'Không tìm thấy'},status=404)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=24*1024*1024:raise ValueError('Yêu cầu quá lớn (tối đa 24 MB)')
            payload=json.loads(self.rfile.read(length))
            if not isinstance(payload,dict):raise ValueError('Yêu cầu không hợp lệ')
            with LOCK:
                if any(j['status']=='running' for j in JOBS.values()):return self.send({'error':'Đang có một lượt phân tích. Vui lòng chờ hoàn tất.'},status=409)
                if len(JOBS)>=30:
                    del JOBS[next(iter(JOBS))]
                job_id=secrets.token_hex(12);JOBS[job_id]={'status':'running','message':'Đang chuẩn bị…'}
            POOL.submit(worker,job_id,payload)
            self.send({'job_id':job_id},status=202)
        except (ValueError,UnicodeError) as exc:self.send({'error':str(exc)},status=400)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8501)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'CaTE-X: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
