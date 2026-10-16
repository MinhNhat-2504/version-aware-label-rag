(() => {
  "use strict";

  const data = window.LABELQA_REVIEW_DATA;
  if (!data || data.schema_version !== "1.0" || data.model_outputs_included !== false) {
    document.body.innerHTML = "<main class='workspace'><h1>Không thể mở gói thẩm định</h1><p>Dữ liệu thiếu, sai phiên bản hoặc chứa trường đầu ra mô hình.</p></main>";
    throw new Error("Invalid reviewer data bundle");
  }

  const packetHash = data.inputs_sha256["data/review/version_holdout_reader_a_blank.jsonl"];
  const storageKey = `labelqa-review:${packetHash}`;
  const pairs = data.pairs;
  const items = pairs.flatMap(pair => pair.items);
  const itemByQid = new Map(items.map(item => [item.qid, item]));
  const fixedByQid = new Map(data.packet_rows.map(row => [row.qid, row]));
  const identityFields = data.identity_fields;
  const annotationFields = data.annotation_fields;
  const activeClaim = {};
  const sourceView = {};
  const lastSelection = {};
  let currentPair = 0;
  let toastTimer;

  const dom = {
    reviewerId: document.querySelector("#reviewer-id"),
    progressText: document.querySelector("#progress-text"),
    progressBar: document.querySelector("#progress-bar"),
    pairList: document.querySelector("#pair-list"),
    pairPosition: document.querySelector("#pair-position"),
    drugTitle: document.querySelector("#drug-title"),
    question: document.querySelector("#question"),
    versions: document.querySelector("#versions"),
    toast: document.querySelector("#toast"),
    dialog: document.querySelector("#validation-dialog"),
    dialogSummary: document.querySelector("#validation-summary"),
    dialogErrors: document.querySelector("#validation-errors"),
  };

  function blankReview(row) {
    return {
      reviewer_id: "",
      item_decision: row.item_decision || "",
      independent_answer: row.independent_answer || "",
      answerable_from_version: row.answerable_from_version ?? null,
      required_claims: Array.isArray(row.required_claims) ? row.required_claims : [],
      acceptable_section_indices_per_claim: Array.isArray(row.acceptable_section_indices_per_claim) ? row.acceptable_section_indices_per_claim : [],
      version_change_judgment: row.version_change_judgment || "",
      clinical_significance: row.clinical_significance || "",
      notes: row.notes || "",
      acceptable_evidence_spans_per_claim: Array.isArray(row.acceptable_evidence_spans_per_claim) ? row.acceptable_evidence_spans_per_claim : [],
    };
  }

  function parallelToClaims(review) {
    return (review.required_claims || []).map((text, index) => ({
      text,
      evidence: structuredClone((review.acceptable_evidence_spans_per_claim || [])[index] || []),
    }));
  }

  function makeInitialState() {
    const reviews = {};
    for (const row of data.packet_rows) {
      const review = blankReview(row);
      reviews[row.qid] = { ...review, claims: parallelToClaims(review) };
    }
    return { dataset_hash: packetHash, reviewer_id: "", reviews };
  }

  function loadLocal() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      if (!saved || saved.dataset_hash !== packetHash || !saved.reviews) return makeInitialState();
      const fresh = makeInitialState();
      fresh.reviewer_id = typeof saved.reviewer_id === "string" ? saved.reviewer_id : "";
      for (const qid of Object.keys(fresh.reviews)) {
        if (saved.reviews[qid]) fresh.reviews[qid] = { ...fresh.reviews[qid], ...saved.reviews[qid] };
      }
      return fresh;
    } catch (_) {
      return makeInitialState();
    }
  }

  let state = loadLocal();

  function save() {
    localStorage.setItem(storageKey, JSON.stringify(state));
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    })[char]);
  }

  function normalize(value) {
    return String(value ?? "").trim().replace(/\s+/g, " ").toLocaleLowerCase();
  }

  function showToast(message) {
    dom.toast.textContent = message;
    dom.toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => dom.toast.classList.remove("show"), 2800);
  }

  function reviewFor(qid) {
    return state.reviews[qid];
  }

  function sectionMap(item) {
    return new Map(item.sections.map(section => [section.section_index, section.text]));
  }

  function itemErrors(item) {
    const review = reviewFor(item.qid);
    const errors = [];
    if (!state.reviewer_id.trim()) errors.push("Chưa nhập mã reviewer.");
    if (!["accept", "rewrite", "reject"].includes(review.item_decision)) {
      errors.push("Chưa chọn quyết định cho câu hỏi.");
      return errors;
    }
    if (review.item_decision !== "accept") return errors;
    if (!["material", "text_only", "stable", "uncertain"].includes(review.version_change_judgment)) {
      errors.push("Chưa đánh giá thay đổi phiên bản.");
    }
    if (typeof review.answerable_from_version !== "boolean") errors.push("Chưa đánh giá khả năng trả lời từ phiên bản này.");
    if (!review.independent_answer.trim()) errors.push("Câu trả lời độc lập đang trống.");
    if (review.answerable_from_version === false) {
      if (review.claims.length) errors.push("Mục không trả lời được phải không có claim hoặc bằng chứng dương tính.");
      return errors;
    }
    if (review.answerable_from_version !== true) return errors;
    if (!review.claims.length) errors.push("Cần ít nhất một claim nguyên tử.");
    const sections = sectionMap(item);
    review.claims.forEach((claim, claimIndex) => {
      if (!String(claim.text || "").trim()) errors.push(`Claim ${claimIndex + 1} đang trống.`);
      if (!Array.isArray(claim.evidence) || !claim.evidence.length) {
        errors.push(`Claim ${claimIndex + 1} chưa có đoạn bằng chứng.`);
        return;
      }
      claim.evidence.forEach((span, spanIndex) => {
        const prefix = `Claim ${claimIndex + 1}, bằng chứng ${spanIndex + 1}`;
        if (!Number.isInteger(span.section_index) || !sections.has(span.section_index)) {
          errors.push(`${prefix}: section không tồn tại.`);
        } else if (!String(span.quote || "").trim() || !normalize(sections.get(span.section_index)).includes(normalize(span.quote))) {
          errors.push(`${prefix}: trích dẫn không khớp nguyên văn nguồn đóng băng.`);
        }
        if (!["prose", "table_row"].includes(span.source_kind)) errors.push(`${prefix}: loại nguồn không hợp lệ.`);
        if (span.source_kind === "table_row") {
          if (!Number.isInteger(span.table_index) || span.table_index < 0 || !Number.isInteger(span.row_index) || span.row_index < 0) {
            errors.push(`${prefix}: cần chỉ số bảng và hàng là số nguyên không âm.`);
          }
        } else if (span.table_index != null || span.row_index != null) {
          errors.push(`${prefix}: nguồn prose không được có tọa độ bảng.`);
        }
      });
    });
    return errors;
  }

  function validateAll() {
    return items.flatMap(item => itemErrors(item).map(message => ({ qid: item.qid, message })));
  }

  function itemIsValid(item) {
    return itemErrors(item).length === 0;
  }

  function pairIsComplete(pair) {
    return pair.items.every(itemIsValid);
  }

  function renderProgress() {
    const valid = items.filter(itemIsValid).length;
    dom.progressText.textContent = `${valid} / ${items.length}`;
    dom.progressBar.style.width = `${100 * valid / items.length}%`;
  }

  function renderSidebar() {
    dom.pairList.innerHTML = pairs.map((pair, index) => {
      const done = pair.items.filter(itemIsValid).length;
      return `<li><button type="button" data-pair="${index}" class="${pairIsComplete(pair) ? "complete" : ""}" aria-current="${index === currentPair}">
        <span class="pair-number">${pairIsComplete(pair) ? "✓" : index + 1}</span>
        <span class="pair-name">${escapeHtml(pair.title)}</span>
        <span class="pair-status">${done}/2</span>
      </button></li>`;
    }).join("");
    renderProgress();
  }

  function selectOptions(options, value, placeholder) {
    return `<option value="">${escapeHtml(placeholder)}</option>` + options.map(([key, label]) =>
      `<option value="${key}" ${value === key ? "selected" : ""}>${escapeHtml(label)}</option>`
    ).join("");
  }

  function evidenceLabel(span) {
    const where = span.source_kind === "table_row"
      ? `§${span.section_index} · bảng ${span.table_index}, hàng ${span.row_index}`
      : `§${span.section_index} · văn xuôi`;
    return `<span>${escapeHtml(where)}</span><q>${escapeHtml(span.quote)}</q>`;
  }

  function claimsHtml(item, review) {
    if (!review.claims.length) return `<p class="empty-state">Chưa có claim. Mỗi claim chỉ nên chứa một sự kiện có thể kiểm tra.</p>`;
    return review.claims.map((claim, index) => `<article class="claim ${activeClaim[item.qid] === index ? "active" : ""}" data-claim="${index}">
      <div class="claim-top">
        <button class="claim-index" type="button" data-action="activate-claim" data-index="${index}" aria-label="Chọn claim ${index + 1}">${index + 1}</button>
        <input data-action="claim-text" data-index="${index}" value="${escapeHtml(claim.text)}" aria-label="Nội dung claim ${index + 1}" placeholder="Một sự kiện nguyên tử cần có trong đáp án">
        <button class="remove-button" type="button" data-action="remove-claim" data-index="${index}" aria-label="Xóa claim ${index + 1}">×</button>
      </div>
      <div class="evidence-list">
        ${claim.evidence.length ? claim.evidence.map((span, spanIndex) => `<div class="evidence-chip">${evidenceLabel(span)}<button class="remove-button" type="button" data-action="remove-evidence" data-index="${index}" data-span="${spanIndex}" aria-label="Xóa bằng chứng">×</button></div>`).join("") : `<span class="empty-state">Chưa có bằng chứng.</span>`}
      </div>
    </article>`).join("");
  }

  function versionCard(item, side) {
    const review = reviewFor(item.qid);
    const accepted = review.item_decision === "accept";
    const answerable = review.answerable_from_version === true;
    const selection = lastSelection[item.qid];
    const errors = itemErrors(item);
    return `<article class="version-card" data-qid="${escapeHtml(item.qid)}">
      <header class="version-header">
        <h3><span>${side === 0 ? "Phiên bản trước" : "Phiên bản sau"}</span><span>v${item.spl_version}</span></h3>
        <p>${escapeHtml(item.published_date)} · SHA-256 <span class="hash">${escapeHtml(item.xml_sha256.slice(0, 12))}…</span></p>
        <a class="source-link" href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener noreferrer">Mở đúng phiên bản nguồn DailyMed ↗</a>
      </header>
      <div class="review-form">
        <div class="field-grid">
          <div class="field"><label class="required" for="decision-${item.qid}">Quyết định câu hỏi</label>
            <select id="decision-${item.qid}" data-field="item_decision">${selectOptions([["accept", "Accept — dùng nguyên câu hỏi"], ["rewrite", "Rewrite — cần viết lại"], ["reject", "Reject — loại khỏi bộ test"]], review.item_decision, "Chọn quyết định")}</select>
          </div>
          <div class="field"><label class="required" for="change-${item.qid}">Thay đổi phiên bản</label>
            <select id="change-${item.qid}" data-field="version_change_judgment" ${accepted ? "" : "disabled"}>${selectOptions([["material", "Material — thay đổi nội dung"], ["text_only", "Text only — chỉ cách viết"], ["stable", "Stable — không đổi"], ["uncertain", "Uncertain — chưa chắc"]], review.version_change_judgment, "Chọn đánh giá")}</select>
          </div>
        </div>
        <div class="field conditional" ${accepted ? "" : "hidden"}><label class="required" for="answerable-${item.qid}">Nguồn phiên bản này có trả lời được?</label>
          <select id="answerable-${item.qid}" data-field="answerable_from_version">${selectOptions([["true", "Có"], ["false", "Không"]], review.answerable_from_version === null ? "" : String(review.answerable_from_version), "Chọn khả năng trả lời")}</select>
        </div>
        <div class="field conditional" ${accepted ? "" : "hidden"}><label class="required" for="answer-${item.qid}">Câu trả lời độc lập</label>
          <textarea id="answer-${item.qid}" data-field="independent_answer" placeholder="Trả lời chỉ dựa trên phiên bản đang đọc">${escapeHtml(review.independent_answer)}</textarea>
        </div>
        <div class="field"><label for="clinical-${item.qid}">Ý nghĩa lâm sàng</label>
          <textarea id="clinical-${item.qid}" data-field="clinical_significance" placeholder="Ghi ngắn gọn; có thể để trống nếu chưa xác định">${escapeHtml(review.clinical_significance)}</textarea>
        </div>
        <div class="field"><label for="notes-${item.qid}">Ghi chú / đề xuất viết lại</label>
          <textarea id="notes-${item.qid}" data-field="notes" placeholder="Với Rewrite hoặc Reject, ghi lý do và câu hỏi đề xuất tại đây">${escapeHtml(review.notes)}</textarea>
        </div>
      </div>
      <section class="claim-zone conditional" ${accepted && answerable ? "" : "hidden"}>
        <div class="section-heading"><h4>Claims nguyên tử và bằng chứng</h4><button class="small-button" type="button" data-action="add-claim">+ Thêm claim</button></div>
        <div class="claims">${claimsHtml(item, review)}</div>
      </section>
      <section class="source-zone">
        <div class="source-tools">
          <div class="section-heading"><h4>Nguồn đóng băng · ${item.sections.length} sections</h4><span class="help">Không xếp hạng</span></div>
          <div class="source-tools-row"><input class="source-search" type="search" data-action="search" placeholder="Lọc theo tiêu đề hoặc nội dung section" value="${escapeHtml(sourceView[item.qid]?.query || "")}"><button class="small-button" type="button" data-action="clear-search">Xóa lọc</button></div>
          <p class="source-count" data-role="source-count"></p>
          <p class="table-notice">Văn bản bảng trong snapshot đã được làm phẳng. Với bằng chứng hàng bảng, hãy đối chiếu bố cục ở liên kết đúng phiên bản phía trên trước khi ghi tọa độ.</p>
        </div>
        <div class="selection-tray conditional" ${accepted && answerable ? "" : "hidden"}>
          <h4>Đoạn đang chọn ${activeClaim[item.qid] == null ? "· hãy chọn một claim" : `· gắn vào claim ${activeClaim[item.qid] + 1}`}</h4>
          <p class="selected-quote">${selection ? `§${selection.section_index}: “${escapeHtml(selection.quote)}”` : "Bôi đen nguyên văn trong một section bên dưới."}</p>
          <div class="evidence-controls">
            <label>Loại nguồn<select data-role="source-kind"><option value="prose">Văn xuôi</option><option value="table_row">Hàng bảng</option></select></label>
            <label>Chỉ số bảng<input data-role="table-index" type="number" min="0" step="1" disabled></label>
            <label>Chỉ số hàng<input data-role="row-index" type="number" min="0" step="1" disabled></label>
            <button class="primary-button" type="button" data-action="add-evidence">Thêm bằng chứng</button>
          </div>
          <p class="help">Với hàng bảng, đối chiếu tọa độ bằng XML nguồn sau khi được mở khóa. Trích dẫn vẫn phải khớp văn bản snapshot.</p>
        </div>
        <div class="sections" data-role="sections"></div>
        <button class="secondary-button load-more" type="button" data-action="load-more" hidden>Xem thêm sections</button>
      </section>
      <p class="item-error ${errors.length ? "" : "item-valid"}">${errors.length ? escapeHtml(errors[0]) : "✓ Mục này đã hợp lệ."}</p>
    </article>`;
  }

  function matchingSections(item) {
    const query = normalize(sourceView[item.qid]?.query || "");
    if (!query) return item.sections;
    return item.sections.filter(section => normalize(`${section.title} ${section.code} ${section.text}`).includes(query));
  }

  function renderSections(item) {
    const card = dom.versions.querySelector(`[data-qid="${CSS.escape(item.qid)}"]`);
    if (!card) return;
    const view = sourceView[item.qid] || (sourceView[item.qid] = { query: "", limit: 8 });
    const matched = matchingSections(item);
    const shown = matched.slice(0, view.limit);
    card.querySelector("[data-role='source-count']").textContent = `${matched.length} section khớp · đang hiển thị ${shown.length}`;
    card.querySelector("[data-role='sections']").innerHTML = shown.map(section => `<details class="source-section" data-section="${section.section_index}">
      <summary>§${section.section_index} · ${escapeHtml(section.title)} <span class="section-code">${escapeHtml(section.code || "")}</span></summary>
      <p class="section-text">${escapeHtml(section.text)}</p>
    </details>`).join("") || `<p class="empty-state">Không có section chứa chuỗi tìm kiếm này.</p>`;
    const more = card.querySelector("[data-action='load-more']");
    more.hidden = shown.length >= matched.length;
    more.textContent = `Xem thêm (${matched.length - shown.length} còn lại)`;
  }

  function renderPair() {
    const pair = pairs[currentPair];
    dom.pairPosition.textContent = `Cặp ${currentPair + 1} / ${pairs.length}`;
    dom.drugTitle.textContent = pair.title;
    dom.question.textContent = pair.question;
    dom.versions.innerHTML = pair.items.map(versionCard).join("");
    pair.items.forEach(renderSections);
    document.querySelector("#previous-pair").disabled = currentPair === 0;
    document.querySelector("#next-pair").disabled = currentPair === pairs.length - 1;
    renderSidebar();
  }

  function rerenderCard(qid) {
    const pair = pairs[currentPair];
    const item = pair.items.find(candidate => candidate.qid === qid);
    if (!item) return;
    const old = dom.versions.querySelector(`[data-qid="${CSS.escape(qid)}"]`);
    const holder = document.createElement("div");
    holder.innerHTML = versionCard(item, pair.items.indexOf(item));
    old.replaceWith(holder.firstElementChild);
    renderSections(item);
    renderSidebar();
  }

  function setPair(index, scroll = true) {
    currentPair = Math.max(0, Math.min(pairs.length - 1, index));
    renderPair();
    if (scroll) document.querySelector("#workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function makeExportRows() {
    return data.packet_rows.map(fixed => {
      const review = reviewFor(fixed.qid);
      const claims = review.claims || [];
      const spans = claims.map(claim => structuredClone(claim.evidence || []));
      const sections = spans.map(group => [...new Set(group.map(span => span.section_index))]);
      return {
        ...structuredClone(fixed),
        reviewer_id: state.reviewer_id.trim(),
        item_decision: review.item_decision,
        independent_answer: review.independent_answer,
        answerable_from_version: review.answerable_from_version,
        required_claims: claims.map(claim => String(claim.text || "").trim()),
        acceptable_section_indices_per_claim: sections,
        version_change_judgment: review.version_change_judgment,
        clinical_significance: review.clinical_significance,
        notes: review.notes,
        acceptable_evidence_spans_per_claim: spans,
      };
    });
  }

  function download(name, content, type) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([content], { type }));
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  }

  function showValidation(errors) {
    dom.dialogSummary.textContent = `Còn ${errors.length} lỗi trên ${new Set(errors.map(error => error.qid)).size} mục. Chọn một lỗi để đi đến đúng cặp.`;
    dom.dialogErrors.innerHTML = errors.slice(0, 80).map(error => `<li><button type="button" data-qid="${escapeHtml(error.qid)}"><strong>${escapeHtml(error.qid)}</strong> — ${escapeHtml(error.message)}</button></li>`).join("");
    dom.dialog.showModal();
  }

  function copyImportedRows(rows) {
    if (!Array.isArray(rows) || rows.length !== data.packet_rows.length) throw new Error(`Cần đúng ${data.packet_rows.length} dòng JSONL.`);
    const byQid = new Map(rows.map(row => [row.qid, row]));
    if (byQid.size !== rows.length || byQid.size !== fixedByQid.size) throw new Error("QID thiếu hoặc bị trùng.");
    const reviewerIds = new Set();
    for (const [qid, fixed] of fixedByQid) {
      const row = byQid.get(qid);
      if (!row) throw new Error(`Thiếu ${qid}.`);
      for (const field of identityFields) {
        if (JSON.stringify(row[field]) !== JSON.stringify(fixed[field])) throw new Error(`Trường nguồn ${field} của ${qid} đã thay đổi.`);
      }
      reviewerIds.add(String(row.reviewer_id || "").trim());
      const review = blankReview(row);
      state.reviews[qid] = { ...review, claims: parallelToClaims(review) };
    }
    if (reviewerIds.size > 1) throw new Error("File chứa nhiều mã reviewer.");
    state.reviewer_id = [...reviewerIds][0] || "";
  }

  function importContent(text) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (_) {
      parsed = text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
    }
    if (parsed && parsed.format === "labelqa-review-draft-v1") {
      if (parsed.dataset_hash !== packetHash || !parsed.state?.reviews) throw new Error("Bản nháp thuộc gói dữ liệu khác.");
      const candidate = parsed.state;
      const rows = data.packet_rows.map(fixed => {
        const review = candidate.reviews[fixed.qid];
        if (!review) throw new Error(`Bản nháp thiếu ${fixed.qid}.`);
        const spans = (review.claims || []).map(claim => claim.evidence || []);
        return { ...fixed, ...review, reviewer_id: candidate.reviewer_id || "",
          required_claims: (review.claims || []).map(claim => claim.text || ""),
          acceptable_section_indices_per_claim: spans.map(group => [...new Set(group.map(span => span.section_index))]),
          acceptable_evidence_spans_per_claim: spans };
      });
      copyImportedRows(rows);
    } else {
      copyImportedRows(parsed);
    }
    save();
    dom.reviewerId.value = state.reviewer_id;
    renderPair();
  }

  dom.reviewerId.value = state.reviewer_id;
  dom.reviewerId.addEventListener("input", event => {
    state.reviewer_id = event.target.value;
    save();
    renderSidebar();
  });

  dom.pairList.addEventListener("click", event => {
    const button = event.target.closest("[data-pair]");
    if (button) setPair(Number(button.dataset.pair));
  });
  document.querySelector("#previous-pair").addEventListener("click", () => setPair(currentPair - 1));
  document.querySelector("#next-pair").addEventListener("click", () => setPair(currentPair + 1));
  document.querySelector("#toggle-list").addEventListener("click", event => {
    const collapsed = dom.pairList.classList.toggle("collapsed");
    event.target.textContent = collapsed ? "Mở rộng" : "Thu gọn";
  });

  dom.versions.addEventListener("input", event => {
    const card = event.target.closest("[data-qid]");
    if (!card) return;
    const review = reviewFor(card.dataset.qid);
    if (event.target.dataset.field) {
      review[event.target.dataset.field] = event.target.value;
      save();
      renderProgress();
    }
    if (event.target.dataset.action === "claim-text") {
      review.claims[Number(event.target.dataset.index)].text = event.target.value;
      save();
      renderProgress();
    }
    if (event.target.dataset.action === "search") {
      const qid = card.dataset.qid;
      sourceView[qid] = { query: event.target.value, limit: 8 };
      renderSections(itemByQid.get(qid));
    }
  });

  dom.versions.addEventListener("change", event => {
    const card = event.target.closest("[data-qid]");
    if (!card) return;
    const qid = card.dataset.qid;
    const review = reviewFor(qid);
    if (event.target.dataset.field === "answerable_from_version") {
      review.answerable_from_version = event.target.value === "" ? null : event.target.value === "true";
      if (review.answerable_from_version === false) review.claims = [];
      save();
      rerenderCard(qid);
    } else if (event.target.dataset.field === "item_decision") {
      review.item_decision = event.target.value;
      save();
      rerenderCard(qid);
    } else if (event.target.dataset.role === "source-kind") {
      const table = event.target.value === "table_row";
      card.querySelector("[data-role='table-index']").disabled = !table;
      card.querySelector("[data-role='row-index']").disabled = !table;
    }
  });

  dom.versions.addEventListener("mouseup", event => {
    const paragraph = event.target.closest(".section-text");
    if (!paragraph) return;
    const selection = window.getSelection();
    const quote = selection ? selection.toString().trim() : "";
    if (!quote || !paragraph.contains(selection.anchorNode) || !paragraph.contains(selection.focusNode)) return;
    const card = paragraph.closest("[data-qid]");
    const details = paragraph.closest("[data-section]");
    lastSelection[card.dataset.qid] = { section_index: Number(details.dataset.section), quote };
    const tray = card.querySelector(".selection-tray");
    if (tray && !tray.hidden) {
      tray.querySelector(".selected-quote").textContent = `§${details.dataset.section}: “${quote}”`;
      showToast("Đã giữ đoạn bôi đen. Chọn loại nguồn rồi thêm vào claim.");
    }
  });

  dom.versions.addEventListener("click", event => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const card = button.closest("[data-qid]");
    const qid = card.dataset.qid;
    const review = reviewFor(qid);
    const action = button.dataset.action;
    if (action === "add-claim") {
      review.claims.push({ text: "", evidence: [] });
      activeClaim[qid] = review.claims.length - 1;
      save(); rerenderCard(qid);
    } else if (action === "activate-claim") {
      activeClaim[qid] = Number(button.dataset.index);
      rerenderCard(qid);
    } else if (action === "remove-claim") {
      review.claims.splice(Number(button.dataset.index), 1);
      activeClaim[qid] = review.claims.length ? Math.min(activeClaim[qid] || 0, review.claims.length - 1) : null;
      save(); rerenderCard(qid);
    } else if (action === "remove-evidence") {
      review.claims[Number(button.dataset.index)].evidence.splice(Number(button.dataset.span), 1);
      save(); rerenderCard(qid);
    } else if (action === "add-evidence") {
      const claimIndex = activeClaim[qid];
      const selected = lastSelection[qid];
      if (claimIndex == null || !review.claims[claimIndex]) return showToast("Hãy tạo và chọn một claim trước.");
      if (!selected) return showToast("Hãy bôi đen một đoạn nguyên văn trong section trước.");
      const kind = card.querySelector("[data-role='source-kind']").value;
      const span = { section_index: selected.section_index, quote: selected.quote, source_kind: kind, table_index: null, row_index: null };
      if (kind === "table_row") {
        const table = card.querySelector("[data-role='table-index']").value;
        const row = card.querySelector("[data-role='row-index']").value;
        if (table === "" || row === "" || Number(table) < 0 || Number(row) < 0 || !Number.isInteger(Number(table)) || !Number.isInteger(Number(row))) {
          return showToast("Nhập chỉ số bảng và hàng là số nguyên không âm.");
        }
        span.table_index = Number(table); span.row_index = Number(row);
      }
      review.claims[claimIndex].evidence.push(span);
      lastSelection[qid] = null;
      save(); rerenderCard(qid);
    } else if (action === "clear-search") {
      sourceView[qid] = { query: "", limit: 8 };
      card.querySelector("[data-action='search']").value = "";
      renderSections(itemByQid.get(qid));
    } else if (action === "load-more") {
      const view = sourceView[qid] || (sourceView[qid] = { query: "", limit: 8 });
      view.limit += 12;
      renderSections(itemByQid.get(qid));
    }
  });

  document.querySelector("#export-jsonl").addEventListener("click", () => {
    const errors = validateAll();
    if (errors.length) return showValidation(errors);
    const content = makeExportRows().map(row => JSON.stringify(row)).join("\n") + "\n";
    download(`version_holdout_${state.reviewer_id.trim()}.jsonl`, content, "application/x-ndjson;charset=utf-8");
    showToast("Đã xuất 24 dòng JSONL hợp lệ.");
  });

  document.querySelector("#backup-json").addEventListener("click", () => {
    const backup = { format: "labelqa-review-draft-v1", dataset_hash: packetHash, state };
    download("labelqa_review_draft.json", JSON.stringify(backup, null, 2) + "\n", "application/json;charset=utf-8");
    showToast("Đã tải bản nháp; file này có thể nhập lại vào giao diện.");
  });

  document.querySelector("#import-file").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      importContent(await file.text());
      showToast("Đã nhập và kiểm tra định danh gói dữ liệu.");
    } catch (error) {
      showToast(`Không thể nhập: ${error.message}`);
    } finally {
      event.target.value = "";
    }
  });

  document.querySelector("#clear-local").addEventListener("click", () => {
    if (!window.confirm("Xóa toàn bộ tiến độ đang lưu trong trình duyệt cho gói này?")) return;
    localStorage.removeItem(storageKey);
    state = makeInitialState();
    dom.reviewerId.value = "";
    setPair(0, false);
    showToast("Đã xóa bản lưu cục bộ.");
  });

  dom.dialogErrors.addEventListener("click", event => {
    const button = event.target.closest("[data-qid]");
    if (!button) return;
    const index = pairs.findIndex(pair => pair.items.some(item => item.qid === button.dataset.qid));
    dom.dialog.close();
    setPair(index);
  });

  window.addEventListener("beforeunload", save);
  renderPair();
})();
