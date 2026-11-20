'use strict';
const $=id=>document.getElementById(id);let tab='pair',caps=null,busy=false;
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=x=>Number.isFinite(x)?x.toFixed(4):'—';
const safeURL=u=>{try{const v=new URL(u);return ['http:','https:'].includes(v.protocol)?escapeHTML(v.href):'';}catch{return '';}};
async function api(path,options={}){const res=await fetch(path,options);const data=await res.json();if(!res.ok)throw Error(data.error||'Lỗi kết nối');return data;}
async function refresh(){try{caps=await api('/api/status');const names=['Bước 0 · AI-touched','Tầng 1 · TF-IDF','Tầng 2 · Pretrained','Tầng 3 · Triplet','Tầng 4 · Cross-Encoder','Tầng 5 · LLM'];$('tiers').innerHTML=names.map((n,i)=>`<span class="tier ${caps.tiers[i].ready?'ready':''}" title="${escapeHTML(caps.tiers[i].detail)}"><span class="dot"></span>${n}</span>`).join('');$('ready-label').textContent=caps.tiers[0].ready&&caps.tiers[1].ready&&caps.tiers[3].ready&&caps.tiers[4].ready?'Checkpoint và tầng 1 đã sẵn sàng':'Đang thiếu thành phần pipeline';const componentLines=names.map((name,i)=>`${name}: ${caps.tiers[i].detail}`);componentLines.push(`Fusion: ${caps.fusion_detail}`,`Web Search: ${caps.web?'đã cấu hình':'chưa có Google Search API key / Search Engine ID'}`,'Cặp văn bản cần đủ Tầng 1–5 và fusion.','Tài liệu cần thêm Bước 0 và Web Search.');$('requirements').textContent=componentLines.join('\n');$('status-note').textContent=caps.notice;update();}catch(e){showError(e.message);}}
function update(){const normal=$('mode').value==='normal';$('pair-panel').hidden=tab!=='pair';$('document-panel').hidden=tab!=='document';$('title').innerHTML=tab==='pair'?'Hai cách diễn đạt.<br>Một nội dung?':'Từ tài liệu đến<br>bằng chứng đối chiếu.';$('intro').textContent=tab==='pair'?'Đưa hai đoạn văn vào để phân tích quan hệ paraphrase.':'Tách câu, tìm nguồn ứng viên trên web và tạo báo cáo highlight.';const ready=caps&&(tab==='pair'?caps.full_pair:caps.full_document);const missing=caps?(tab==='pair'?caps.full_pair_missing:caps.full_document_missing):[];$('submit').disabled=busy||!ready;$('mode').disabled=busy;$('submit').textContent=busy?'Đang phân tích…':'Chạy đủ 5 tầng ↗';$('availability').textContent=busy?'Đang xử lý…':ready?'Sẵn sàng · Cả hai chế độ đều dùng đủ 5 tầng':'Thiếu: '+missing.join(' · ');$('mode-note').textContent=tab==='pair'?'Với cặp câu đã có sẵn, cả hai chế độ đều chạy đủ 5 tầng. Chế độ thường/sâu chỉ thay phạm vi tìm nguồn khi kiểm tra tài liệu.':normal?'Kiểm tra thường: Bước 0 chọn câu cần Web Search. Mỗi cặp câu và nguồn ứng viên vẫn chạy đủ 5 tầng.':'Kiểm tra sâu: Web Search áp dụng cho mọi câu. Mỗi cặp câu và nguồn ứng viên chạy đủ 5 tầng.';}
function showError(message){$('error').textContent=message;$('error').hidden=false;}
async function inputValue(prefix){const file=$(prefix+'-file').files[0];if(!file)return {text:$(prefix+'-text').value};if(file.size>8*1024*1024)throw Error('File vượt giới hạn 8 MB');const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=()=>reject(Error('Không đọc được file'));r.readAsDataURL(file);});return {file:{name:file.name,data}};}
function highlightSentences(text, rows) {
  // Backend dùng vị trí theo mã Unicode. Array.from giữ đúng vị trí cả sau emoji.
  const chars = Array.from(text);
  let html = '', cursor = 0;
  for (const row of rows) {
    if (row.status !== 'suspicious') continue;
    html += escapeHTML(chars.slice(cursor, row.start).join(''));
    const sentence = escapeHTML(chars.slice(row.start, row.end).join(''));
    html += `<mark tabindex="0" data-sentence="${row.sentence_id}">${sentence}</mark>`;
    cursor = row.end;
  }
  return html + escapeHTML(chars.slice(cursor).join(''));
}

function renderEvidence(row, pair) {
  const link = safeURL(row.source_url);
  const sourceLabel = pair ? `Câu nguồn ${row.source_sentence_id + 1}` : 'Nguồn ứng viên';
  const threshold = pair ? ` · ngưỡng ${number(row.threshold_used)}` : '';
  const trace = pair ? (row.trace || []).map(t =>
    `<span>Tầng ${t.tang} · ${t.status === 'ok' ? 'Hoàn tất' : 'Lỗi'} · ${Math.round(t.time_ms)} ms</span>`
  ).join('') : '';
  return `<article id="evidence-${row.sentence_id}">
    <h3>CÂU ${row.sentence_id + 1} · ${escapeHTML(row.label || row.status || 'Chưa đủ bằng chứng')}</h3>
    <p>${escapeHTML(row.sentence)}</p>
    ${row.matched_source_sentence ? `<small>${sourceLabel} · điểm fusion ${number(row.probability)}${threshold}</small><blockquote>${escapeHTML(row.matched_source_sentence)}</blockquote>` : ''}
    ${link ? `<a href="${link}" target="_blank" rel="noopener noreferrer">Mở nguồn ↗</a>` : ''}
    ${row.explanation ? `<p>${escapeHTML(row.explanation)}</p>` : ''}
    ${row.complete === false ? '<small>Chưa hoàn tất mọi nguồn hoặc phần giải thích.</small>' : ''}
    ${trace ? `<details><summary>Trạng thái 5 tầng</summary><div class="trace">${trace}</div></details>` : ''}
  </article>`;
}

function render(result) {
  const pair = result.kind === 'pair';
  const rows = result.results || [];
  if (pair && !result.results) {
    showError('Phiên bản máy chủ đang chấm cả đoạn. Hãy khởi động lại app để đối chiếu theo câu.');
    return;
  }
  const downloads = Object.entries(result.downloads || {}).map(([type, url]) =>
    `<a href="${escapeHTML(url)}" download>Tải ${type.toUpperCase()}</a>`
  ).join('');
  const summary = pair && rows.length === 1
    ? `<strong>${number(rows[0].probability)}</strong><div><p>${escapeHTML(rows[0].label)}</p><small>Điểm fusion của cặp câu · ngưỡng ${number(rows[0].threshold_used)}</small></div>`
    : `<strong>${result.suspicious_sentences}<small>/ ${result.total_sentences} câu</small></strong><div><p>Câu nghi vấn cần xem lại</p></div>`;
  const note = pair ? 'Mỗi câu được đối chiếu với câu nguồn gần nhất theo SBERT. Điểm fusion chưa hiệu chỉnh.' : '';
  const text = highlightSentences(result.original_text, rows);
  const warnings = (result.warnings || []).map(w => `<p>ⓘ ${escapeHTML(w)}</p>`).join('');
  $('results').innerHTML = `<div class="result-top"><h2>Kết quả CaTE-X · đủ 5 tầng</h2><div class="downloads">${downloads}</div></div>
    <div class="score">${summary}</div>${note ? `<p>${note}</p>` : ''}
    <div class="result-grid"><div><div class="document">${text}</div>
    ${pair ? `<details><summary>Đoạn nguồn đã nhập</summary><div class="document">${escapeHTML(result.source)}</div></details>` : ''}</div>
    <div class="evidence">${rows.map(r => renderEvidence(r, pair)).join('')}</div></div>
    <div class="warnings">${warnings}</div>`;
  $('results').hidden = false;
  $('results').scrollIntoView({behavior:'smooth',block:'start'});
  document.querySelectorAll('[data-sentence]').forEach(el => {
    const select = () => {
      document.querySelectorAll('.evidence .selected').forEach(e => e.classList.remove('selected'));
      const target = $('evidence-' + el.dataset.sentence);
      target.classList.add('selected');
      target.scrollIntoView({behavior:'smooth',block:'nearest'});
    };
    el.addEventListener('click', select);
    el.addEventListener('keydown', e => { if (e.key === 'Enter') select(); });
  });
}
$('analysis-form').addEventListener('submit',async e=>{e.preventDefault();if(busy)return;busy=true;$('error').hidden=true;$('results').hidden=true;$('progress').hidden=false;$('progress-text').textContent='Đang chuẩn bị…';update();try{const payload={kind:tab,mode:$('mode').value};if(tab==='pair'){payload.source=$('source').value;payload.suspect=$('suspect').value;}else{payload.document=await inputValue('document');}const job=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':caps.token},body:JSON.stringify(payload)});for(;;){await new Promise(r=>setTimeout(r,1300));const status=await api('/api/jobs/'+job.job_id);$('progress-text').textContent=status.message;if(status.status==='error')throw Error(status.message);if(status.status==='done'){render(status.result);break;}}}catch(error){showError(error.message);}finally{busy=false;$('progress').hidden=true;update();}});
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{if(busy)return;tab=b.dataset.tab;document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));$('results').hidden=true;$('error').hidden=true;update();}));$('mode').addEventListener('change',()=>{$('results').hidden=true;update();});$('refresh').addEventListener('click',refresh);for(const id of ['source','suspect'])$(id).addEventListener('input',()=>$(id+'-count').textContent=$(id).value.length+' ký tự');$('example').addEventListener('click',()=>{if(busy)return;const source='Urban trees reduce surface temperatures by providing shade. They also improve air quality by filtering pollutants.';const suspect='Trees in cities cool the ground through shading. Urban vegetation also helps remove pollutants from the air.';if(tab==='pair'){$('source').value=source;$('suspect').value=suspect;for(const id of ['source','suspect'])$(id).dispatchEvent(new Event('input'));}else{$('document-text').value=suspect+' The museum opens at nine every morning.';$('document-file').value='';}});refresh();
