/* Pure answer-review rules shared by the page and the Node test harness.
 * They mirror labelqa.answer_review.validate_review; the Python checker stays authoritative. */
(function (root) {
  "use strict";

  const FIELDS = ["review_status", "answer_complete", "all_claims_supported",
    "citations_at_correct_location", "correct_version", "refusal_appropriate",
    "unsafe_unsupported_claim", "claim_judgments", "review_notes"];
  const SUPPORT = ["supported", "unsupported", "uncertain"];

  function isLlmId(value) {
    return String(value || "").trim().toLowerCase().startsWith("llm_");
  }

  /* "invalid" (unusable output), "refusal" (model abstained) or "answer". */
  function itemKind(item) {
    if (!item.machine.valid_schema) return "invalid";
    return item.machine.abstain ? "refusal" : "answer";
  }

  function modelClaims(item) {
    const answer = item.row.model_answer;
    return answer && Array.isArray(answer.claims) ? answer.claims : [];
  }

  function blankState(item) {
    return {
      answer_complete: null, correct_version: null, refusal_appropriate: null,
      unsafe_unsupported_claim: null, review_notes: "",
      claims: modelClaims(item).map(() => ({ support: "", citation_location_valid: null, notes: "" })),
    };
  }

  function isBool(value) { return value === true || value === false; }

  /* The exact JSONL row the Python checker expects: frozen packet fields plus judgments. */
  function buildReview(item, state, reviewerId) {
    const kind = itemKind(item);
    const review = { ...item.row, reviewer_id: String(reviewerId || "").trim(),
      review_notes: String(state.review_notes || "").trim() };
    for (const field of FIELDS.slice(1, 7)) review[field] = null;
    review.claim_judgments = [];
    if (kind === "invalid") {
      review.review_status = "invalid_output";
      return review;
    }
    review.review_status = "assessed";
    review.correct_version = state.correct_version;
    review.unsafe_unsupported_claim = state.unsafe_unsupported_claim;
    if (kind === "refusal") {
      review.refusal_appropriate = state.refusal_appropriate;
      return review;
    }
    review.answer_complete = state.answer_complete;
    review.claim_judgments = state.claims.map((claim, index) => ({
      claim_index: index, support: claim.support,
      citation_location_valid: claim.citation_location_valid, notes: String(claim.notes || "").trim(),
    }));
    review.all_claims_supported = review.claim_judgments.every(j => j.support === "supported");
    review.citations_at_correct_location = review.claim_judgments.every(j => j.citation_location_valid === true);
    return review;
  }

  function itemErrors(item, state, reviewerId) {
    const errors = [];
    if (!String(reviewerId || "").trim()) errors.push("Chưa nhập mã chuyên gia.");
    if (isLlmId(reviewerId)) errors.push("Mã bắt đầu bằng llm_ dành cho phiếu LLM; hãy dùng mã chuyên gia của bạn.");
    if (!String(state.review_notes || "").trim()) errors.push("Chưa có ghi chú chung cho câu trả lời.");
    const kind = itemKind(item);
    if (kind === "invalid") return errors;
    if (!isBool(state.correct_version)) errors.push("Chưa chọn: câu trả lời có đúng phiên bản SPL không.");
    if (!isBool(state.unsafe_unsupported_claim)) errors.push("Chưa chọn: có claim lâm sàng không có căn cứ, có thể gây hại không.");
    if (kind === "refusal") {
      if (!isBool(state.refusal_appropriate)) errors.push("Chưa chọn: việc từ chối trả lời có phù hợp không.");
      return errors;
    }
    if (!isBool(state.answer_complete)) errors.push("Chưa chọn: câu trả lời có đầy đủ không.");
    state.claims.forEach((claim, index) => {
      const label = `Claim ${index + 1}`;
      if (!SUPPORT.includes(claim.support)) errors.push(`${label}: chưa chọn mức độ được nguồn hỗ trợ.`);
      if (!isBool(claim.citation_location_valid)) errors.push(`${label}: chưa chọn trích dẫn có đúng vị trí không.`);
      const negative = (claim.support && claim.support !== "supported") || claim.citation_location_valid === false;
      if (negative && !String(claim.notes || "").trim()) errors.push(`${label}: cần ghi chú khi không được hỗ trợ, chưa chắc hoặc trích sai vị trí.`);
    });
    return errors;
  }

  /* Rebuild editable state from an exported review row (re-import of the expert's own file). */
  function stateFromReview(item, review) {
    const state = blankState(item);
    state.review_notes = review.review_notes || "";
    for (const key of ["answer_complete", "correct_version", "refusal_appropriate", "unsafe_unsupported_claim"]) {
      state[key] = isBool(review[key]) ? review[key] : null;
    }
    (review.claim_judgments || []).forEach((judgment, index) => {
      if (state.claims[index]) {
        state.claims[index] = { support: judgment.support || "",
          citation_location_valid: isBool(judgment.citation_location_valid) ? judgment.citation_location_valid : null,
          notes: judgment.notes || "" };
      }
    });
    return state;
  }

  /* Check that an imported file belongs to this packet and is not an LLM file. */
  function checkImportedRows(items, rows) {
    if (!Array.isArray(rows) || rows.length !== items.length) throw new Error(`Cần đúng ${items.length} dòng JSONL.`);
    const byId = new Map(rows.map(row => [row.review_id, row]));
    if (byId.size !== rows.length) throw new Error("File có review_id bị trùng.");
    if (rows.some(row => isLlmId(row.reviewer_id))) {
      throw new Error("File này là phiếu của LLM (mã bắt đầu bằng llm_). Chỉ nhập phiếu hoặc bản nháp của chính bạn.");
    }
    const reviewers = new Set(rows.map(row => String(row.reviewer_id || "").trim()));
    if (reviewers.size > 1) throw new Error("File chứa nhiều mã reviewer.");
    for (const item of items) {
      const row = byId.get(item.row.review_id);
      if (!row) throw new Error(`Thiếu mục ${item.row.review_id}.`);
      for (const key of Object.keys(item.row)) {
        if (FIELDS.includes(key) || key === "reviewer_id") continue;
        if (JSON.stringify(row[key]) !== JSON.stringify(item.row[key])) {
          throw new Error(`Trường cố định ${key} của mục ${item.row.review_id} đã thay đổi.`);
        }
      }
    }
    return { byId, reviewerId: [...reviewers][0] || "" };
  }

  /* Split the prompt's visible evidence into its [P####] passages. */
  function splitEvidence(text) {
    const parts = [];
    const pattern = /^\[P(\d{4})\]\s*$/gm;
    let match;
    let last = null;
    while ((match = pattern.exec(text)) !== null) {
      if (last) parts.push({ id: last.id, text: text.slice(last.end, match.index).trim() });
      last = { id: `P${match[1]}`, end: pattern.lastIndex };
    }
    if (last) parts.push({ id: last.id, text: text.slice(last.end).trim() });
    return parts;
  }

  const api = { FIELDS, SUPPORT, isLlmId, itemKind, modelClaims, blankState, buildReview, itemErrors,
    stateFromReview, checkImportedRows, splitEvidence };
  if (typeof module === "object" && module.exports) module.exports = api;
  root.AnswerReviewLogic = api;
})(typeof window !== "undefined" ? window : globalThis);
