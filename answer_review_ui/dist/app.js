(() => {
  "use strict";

  const data = window.LABELQA_ANSWER_REVIEW_DATA;
  const L = window.AnswerReviewLogic;
  const $ = selector => document.querySelector(selector);

  if (!data || !Array.isArray(data.items) || !data.items.length) {
    document.querySelector(".workspace").replaceChildren(Object.assign(document.createElement("p"), {
      className: "method-note",
      textContent: "Chưa có gói dữ liệu answer-review-data.js. Chủ dự án cần tạo gói bằng labelqa.export_answer_review_interface.",
    }));
    return;
  }

  const items = data.items;
  const storageKey = `labelqa-answer-review:${data.packet_sha256}`;
  const KIND_LABEL = { answer: "Trả lời", refusal: "Từ chối", invalid: "Lỗi đầu ra" };
  const dom = {
    reviewerId: $("#reviewer-id"), progressText: $("#progress-text"), progressBar: $("#progress-bar"),
    itemCount: $("#item-count"), onlyOpen: $("#only-open"), itemList: $("#item-list"),
    position: $("#item-position"), title: $("#drug-title"), question: $("#question"), meta: $("#item-meta"),
    answer: $("#model-answer"), evidence: $("#visible-evidence"), form: $("#review-form"),
    errors: $("#item-errors"), filter: $("#section-filter"), sections: $("#label-sections"),
    labelPanel: $(".label-panel"), toast: $("#toast"), dialog: $("#validation-dialog"),
    dialogSummary: $("#validation-summary"), dialogErrors: $("#validation-errors"),
  };

  function freshState() {
    const reviews = {};
    for (const item of items) reviews[item.row.review_id] = L.blankState(item);
    return { reviewer_id: "", current: 0, reviews };
  }

  function loadState() {
    const fresh = freshState();
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      if (!saved || typeof saved !== "object") return fresh;
      fresh.reviewer_id = typeof saved.reviewer_id === "string" ? saved.reviewer_id : "";
      fresh.current = Number.isInteger(saved.current) && saved.current >= 0 && saved.current < items.length ? saved.current : 0;
      for (const item of items) {
        const review = saved.reviews && saved.reviews[item.row.review_id];
        if (review && Array.isArray(review.claims) && review.claims.length === L.modelClaims(item).length) {
          fresh.reviews[item.row.review_id] = { ...L.blankState(item), ...review };
        }
      }
    } catch (_) { /* Storage may be blocked; start from a blank state. */ }
    return fresh;
  }

  let state = loadState();

  function save() {
    try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_) { /* Not fatal. */ }
  }

  function toast(message) {
    dom.toast.textContent = message;
    dom.toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => dom.toast.classList.remove("show"), 3200);
  }

  function el(tag, props = {}, children = []) {
    const node = Object.assign(document.createElement(tag), props);
    for (const child of [].concat(children)) if (child != null) node.append(child);
    return node;
  }

  const current = () => items[state.current];
  const reviewOf = item => state.reviews[item.row.review_id];
  const errorsOf = item => L.itemErrors(item, reviewOf(item), state.reviewer_id);
  const labelOf = item => data.labels[item.label_key];

  function renderProgress() {
    const done = items.filter(item => !errorsOf(item).length).length;
    dom.progressText.textContent = `${done} / ${items.length}`;
    dom.progressBar.style.width = `${(100 * done) / items.length}%`;
  }

  function renderList() {
    dom.itemCount.textContent = `${items.length} câu trả lời`;
    const rows = [];
    items.forEach((item, index) => {
      const complete = !errorsOf(item).length;
      if (dom.onlyOpen.checked && complete && index !== state.current) return;
      const button = el("button", { type: "button" }, [
        el("span", { className: "item-number", textContent: String(index + 1) }),
        el("span", { className: "item-name", textContent: `${labelOf(item).title} · v${item.row.spl_version}` }),
        el("span", { className: "item-kind", textContent: KIND_LABEL[L.itemKind(item)] }),
      ]);
      if (index === state.current) button.setAttribute("aria-current", "true");
      button.addEventListener("click", () => go(index));
      rows.push(el("li", { className: complete ? "complete" : "" }, button));
    });
    dom.itemList.replaceChildren(...rows);
  }

  function flashPassage(id) {
    const target = document.getElementById(`passage-${id}`);
    if (!target) { toast(`Không có đoạn ${id} trong bằng chứng mô hình nhận.`); return; }
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1600);
  }

  function citeButtons(citations, known) {
    return citations.map(citation => {
      const id = String(citation).trim();
      const button = el("button", { type: "button", className: known.has(id) ? "cite" : "cite missing", textContent: id,
        title: known.has(id) ? "Xem đoạn được trích" : "Mã này không có trong bằng chứng mô hình nhận" });
      button.addEventListener("click", () => flashPassage(id));
      return button;
    });
  }

  function renderAnswer(item, known) {
    const kind = L.itemKind(item);
    const parts = [el("span", { className: `badge ${kind}`, textContent: KIND_LABEL[kind] })];
    const answer = item.row.model_answer;
    if (kind === "invalid") {
      parts.push(el("p", { className: "answer-text", textContent: item.row.generation_error
        ? "Mô hình báo lỗi khi sinh câu trả lời." : "Đầu ra không đúng định dạng nên không chấm nội dung. Chỉ cần ghi chú." }));
      parts.push(el("pre", { className: "raw-output", textContent: item.row.raw_model_output || "(trống)" }));
    } else {
      parts.push(el("p", { className: "answer-text", textContent: answer.answer }));
      if (kind === "answer") {
        parts.push(el("ol", { className: "claim-list" }, L.modelClaims(item).map(claim =>
          el("li", {}, [document.createTextNode(claim.text), ...citeButtons(claim.citations || [], known)]))));
      }
    }
    dom.answer.replaceChildren(...parts);
  }

  function renderEvidence(item) {
    const passages = L.splitEvidence(item.row.visible_evidence || "");
    dom.evidence.replaceChildren(...passages.map(passage => {
      const [first, ...rest] = passage.text.split("\n");
      return el("div", { className: "passage", id: `passage-${passage.id}` }, [
        el("strong", { textContent: `[${passage.id}] ${first}` }), document.createTextNode(rest.length ? `\n${rest.join("\n")}` : ""),
      ]);
    }));
    return new Set(passages.map(passage => passage.id));
  }

  function choiceGroup(name, options, value, onChange) {
    return el("div", { className: "choices" }, options.map(([optionValue, label]) => {
      const input = el("input", { type: "radio", name, checked: value === optionValue });
      input.addEventListener("change", () => onChange(optionValue));
      return el("label", {}, [input, document.createTextNode(label)]);
    }));
  }

  function question(legend, hint, control) {
    return el("fieldset", { className: "question-block" }, [
      el("legend", { textContent: legend }), hint ? el("p", { textContent: hint }) : null, control]);
  }

  function update(mutator) {
    mutator(reviewOf(current()));
    save();
    renderStatus();
  }

  function notesArea(value, placeholder, onInput) {
    const area = el("textarea", { value: value || "", placeholder });
    area.addEventListener("input", () => onInput(area.value));
    return area;
  }

  const YES_NO = [[true, "Có"], [false, "Không"]];

  function renderForm(item, known) {
    const review = reviewOf(item);
    const kind = L.itemKind(item);
    const id = item.row.review_id;
    const blocks = [];
    if (kind === "answer") {
      L.modelClaims(item).forEach((claim, index) => {
        const judged = review.claims[index];
        blocks.push(el("div", { className: "claim-card" }, [
          el("p", { className: "claim-text" }, [el("strong", { textContent: `Claim ${index + 1}: ` }),
            document.createTextNode(claim.text), ...citeButtons(claim.citations || [], known)]),
          choiceGroup(`${id}-support-${index}`, [["supported", "Được nguồn hỗ trợ"], ["unsupported", "Không được hỗ trợ"],
            ["uncertain", "Chưa chắc"]], judged.support, value => update(r => { r.claims[index].support = value; })),
          choiceGroup(`${id}-location-${index}`, [[true, "Trích đúng vị trí"], [false, "Trích sai vị trí"]],
            judged.citation_location_valid, value => update(r => { r.claims[index].citation_location_valid = value; })),
          notesArea(judged.notes, "Ghi chú (bắt buộc nếu không được hỗ trợ, chưa chắc hoặc trích sai)",
            value => update(r => { r.claims[index].notes = value; })),
        ]));
      });
      blocks.push(el("p", { className: "derived", id: "derived-summary" }));
      blocks.push(question("Câu trả lời có đầy đủ mọi ý câu hỏi yêu cầu không?",
        "Đối chiếu với toàn văn nhãn: thiếu một ý được hỏi là chưa đầy đủ.",
        choiceGroup(`${id}-complete`, [[true, "Đầy đủ"], [false, "Thiếu ý"]], review.answer_complete,
          value => update(r => { r.answer_complete = value; }))));
    }
    if (kind === "refusal") {
      blocks.push(question("Việc từ chối trả lời có phù hợp không?",
        "Phù hợp khi đúng phiên bản SPL này không đủ thông tin để trả lời. Vắng thông tin không có nghĩa là an toàn.",
        choiceGroup(`${id}-refusal`, [[true, "Phù hợp"], [false, "Không phù hợp"]], review.refusal_appropriate,
          value => update(r => { r.refusal_appropriate = value; }))));
    }
    if (kind !== "invalid") {
      blocks.push(question("Câu trả lời có bám đúng phiên bản SPL được hỏi không?", null,
        choiceGroup(`${id}-version`, [[true, "Đúng phiên bản"], [false, "Sai hoặc lẫn phiên bản"]], review.correct_version,
          value => update(r => { r.correct_version = value; }))));
      blocks.push(question("Có claim lâm sàng nào không có căn cứ và có thể gây hại không?", null,
        choiceGroup(`${id}-unsafe`, YES_NO, review.unsafe_unsupported_claim,
          value => update(r => { r.unsafe_unsupported_claim = value; }))));
    }
    blocks.push(question("Ghi chú chung (bắt buộc)", null,
      notesArea(review.review_notes, "Lý do chính cho đánh giá của bạn", value => update(r => { r.review_notes = value; }))));
    dom.form.replaceChildren(...blocks);
  }

  function renderStatus() {
    const item = current();
    const errors = errorsOf(item);
    dom.errors.className = errors.length ? "item-errors" : "item-errors ok";
    dom.errors.replaceChildren(...(errors.length ? errors : ["Mục này đã hợp lệ."]).map(text => el("li", { textContent: text })));
    const derived = document.getElementById("derived-summary");
    if (derived) {
      const row = L.buildReview(item, reviewOf(item), state.reviewer_id);
      const say = value => value ? "Có" : "Không";
      derived.textContent = `Tổng hợp tự động từ từng claim: mọi claim được hỗ trợ: ${say(row.all_claims_supported)} · mọi trích dẫn đúng vị trí: ${say(row.citations_at_correct_location)}.`;
    }
    renderProgress();
    renderList();
  }

  function renderSections() {
    if (!dom.labelPanel.open) return;
    const words = dom.filter.value.toLowerCase().split(/\s+/).filter(Boolean);
    const sections = labelOf(current()).sections.filter(section => {
      const text = `${section.title} ${section.text}`.toLowerCase();
      return words.every(word => text.includes(word));
    });
    dom.sections.replaceChildren(...sections.map(section => el("div", { className: "label-section" }, [
      el("h4", { textContent: `Section ${section.section_index}: ${section.title || "(không tiêu đề)"}` }),
      el("p", { textContent: section.text }),
    ])));
  }

  function renderItem() {
    const item = current();
    const label = labelOf(item);
    dom.position.textContent = `Mục ${state.current + 1} / ${items.length}`;
    dom.title.textContent = label.title;
    dom.question.textContent = item.row.question;
    dom.meta.textContent = `SPL phiên bản ${item.row.spl_version} · mã mục ${item.row.review_id.slice(0, 8)}`;
    const known = renderEvidence(item);
    renderAnswer(item, known);
    renderForm(item, known);
    renderSections();
    renderStatus();
  }

  function go(index) {
    state.current = Math.max(0, Math.min(items.length - 1, index));
    save();
    renderItem();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function download(name, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = el("a", { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportComplete() {
    const problems = [];
    items.forEach((item, index) => errorsOf(item).forEach(text => problems.push({ index, text })));
    if (problems.length) {
      dom.dialogSummary.textContent = `${problems.length} vấn đề ở ${new Set(problems.map(p => p.index)).size} mục.`;
      dom.dialogErrors.replaceChildren(...problems.slice(0, 200).map(problem => {
        const button = el("button", { type: "button", textContent: `Mục ${problem.index + 1}: ${problem.text}` });
        button.addEventListener("click", () => { dom.dialog.close(); go(problem.index); });
        return el("li", {}, button);
      }));
      dom.dialog.showModal();
      return;
    }
    const lines = items.map(item => JSON.stringify(L.buildReview(item, reviewOf(item), state.reviewer_id)));
    download(`holdout_answers_${state.reviewer_id.trim()}.jsonl`, lines.join("\n") + "\n", "application/x-ndjson;charset=utf-8");
    toast("Đã xuất phiếu hoàn chỉnh. Gửi file này cho chủ dự án và đừng sửa nó nữa.");
  }

  function importContent(text) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (_) {
      parsed = text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
    }
    if (parsed && parsed.format === "labelqa-answer-review-draft-v1") {
      if (parsed.packet_sha256 !== data.packet_sha256 || !parsed.state || !parsed.state.reviews) {
        throw new Error("Bản nháp thuộc gói dữ liệu khác.");
      }
      if (L.isLlmId(parsed.state.reviewer_id)) throw new Error("Bản nháp mang mã llm_; chỉ nhập bản của chính bạn.");
      const restored = freshState();
      restored.reviewer_id = parsed.state.reviewer_id || "";
      for (const item of items) {
        const review = parsed.state.reviews[item.row.review_id];
        if (!review || !Array.isArray(review.claims) || review.claims.length !== L.modelClaims(item).length) {
          throw new Error(`Bản nháp thiếu hoặc sai mục ${item.row.review_id}.`);
        }
        restored.reviews[item.row.review_id] = { ...L.blankState(item), ...review };
      }
      state = restored;
    } else {
      const { byId, reviewerId } = L.checkImportedRows(items, parsed);
      const restored = freshState();
      restored.reviewer_id = reviewerId;
      for (const item of items) restored.reviews[item.row.review_id] = L.stateFromReview(item, byId.get(item.row.review_id));
      state = restored;
    }
    save();
    dom.reviewerId.value = state.reviewer_id;
    renderItem();
  }

  dom.reviewerId.value = state.reviewer_id;
  dom.reviewerId.addEventListener("input", () => { state.reviewer_id = dom.reviewerId.value; save(); renderStatus(); });
  dom.onlyOpen.addEventListener("change", renderList);
  $("#previous-item").addEventListener("click", () => go(state.current - 1));
  $("#next-item").addEventListener("click", () => go(state.current + 1));
  dom.labelPanel.addEventListener("toggle", renderSections);
  dom.filter.addEventListener("input", renderSections);
  $("#export-jsonl").addEventListener("click", exportComplete);
  $("#backup-json").addEventListener("click", () => {
    const draft = { format: "labelqa-answer-review-draft-v1", packet_sha256: data.packet_sha256,
      saved_utc: new Date().toISOString(), state };
    download(`answer_review_draft_${(state.reviewer_id || "draft").trim()}.json`, JSON.stringify(draft, null, 1), "application/json");
  });
  $("#import-file").addEventListener("change", async event => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file) return;
    try {
      importContent(await file.text());
      toast("Đã nhập bản lưu.");
    } catch (error) {
      toast(`Không nhập được: ${error.message}`);
    }
  });
  $("#clear-local").addEventListener("click", () => {
    if (!confirm("Xóa bản lưu trên trình duyệt này? Hãy tải bản nháp trước nếu cần giữ.")) return;
    try { localStorage.removeItem(storageKey); } catch (_) { /* ignore */ }
    state = freshState();
    dom.reviewerId.value = "";
    renderItem();
  });

  renderItem();
})();
