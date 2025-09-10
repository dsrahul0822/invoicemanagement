# app.py
from __future__ import annotations
import csv, os
from typing import TypedDict, Annotated, Literal, Optional, List, Dict
import operator
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# =========================
# ENV & MODEL
# =========================
load_dotenv()
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=LLM_MODEL)

# =========================
# ABSOLUTE CSV PATHS (safe for REPL/Streamlit)
# =========================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INVOICE_CSV = os.getenv("INVOICE_CSV", os.path.join(DATA_DIR, "invoices.csv"))
PO_CSV = os.getenv("PO_CSV", os.path.join(DATA_DIR, "purchase_orders.csv"))

# =========================
# Pydantic Schemas
# =========================
class CreateInvoiceSchema(BaseModel):
    invoice_no: str = Field(..., description="Unique invoice identifier, e.g., INV-0004")
    po_no: str = Field(..., description="Purchase order number, e.g., PO-1001")
    customer: str = Field(..., description="Customer name, e.g., ECMEC")
    amount: float = Field(..., description="Invoice amount as a number")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD format")

    @field_validator("invoice_no", "po_no", "customer", "due_date")
    @classmethod
    def not_blank(cls, v: str):
        if not str(v).strip():
            raise ValueError("Must not be blank")
        return v

class EmailDraft(BaseModel):
    to: str = Field(default="ap@yourcompany.com")
    sender: str = Field(default="noreply@yourcompany.com")
    subject: str
    body: str

# --- Create op parser result ---
class NLParseResult(BaseModel):
    op: Literal["create_invoice"] = "create_invoice"
    fields: CreateInvoiceSchema

# --- Read filters & parser result ---
class ReadInvoiceSchema(BaseModel):
    """Flexible filters for reading invoices via natural language."""
    invoice_no: Optional[str] = None
    po_no: Optional[str] = None
    customer: Optional[str] = None
    due_from: Optional[str] = Field(default=None, description="YYYY-MM-DD inclusive")
    due_to: Optional[str] = Field(default=None, description="YYYY-MM-DD inclusive")

    @field_validator("invoice_no", "po_no", "customer", "due_from", "due_to")
    @classmethod
    def empty_to_none(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v if v else None

class NLReadParseResult(BaseModel):
    op: Literal["read_invoice"] = "read_invoice"
    filters: ReadInvoiceSchema

# --- Update schema & parser result ---
class UpdateInvoiceSchema(BaseModel):
    """Identify target invoice by invoice_no; allow updates to specific fields. Invoice_no itself is not changed."""
    invoice_no: str = Field(..., description="Target invoice number to update, e.g., INV-0004")
    new_po_no: Optional[str] = None
    new_customer: Optional[str] = None
    new_amount: Optional[float] = None
    new_due_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")

    @field_validator("invoice_no")
    @classmethod
    def inv_not_blank(cls, v: str):
        if not str(v).strip():
            raise ValueError("invoice_no cannot be blank")
        return v

    @field_validator("new_po_no", "new_customer", "new_due_date")
    @classmethod
    def strip_or_none(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

class NLUpdateParseResult(BaseModel):
    op: Literal["update_invoice"] = "update_invoice"
    fields: UpdateInvoiceSchema

# --- Delete parsers ---
class DeleteInvoiceSchema(BaseModel):
    invoice_no: str = Field(..., description="Invoice number to delete, e.g., INV-0004")

    @field_validator("invoice_no")
    @classmethod
    def inv_not_blank(cls, v: str):
        if not str(v).strip():
            raise ValueError("invoice_no cannot be blank")
        return v

class NLDeleteInvoiceParseResult(BaseModel):
    op: Literal["delete_invoice"] = "delete_invoice"
    fields: DeleteInvoiceSchema

class DeletePOSchema(BaseModel):
    po_no: str = Field(..., description="PO number to delete, e.g., PO-1001")

    @field_validator("po_no")
    @classmethod
    def po_not_blank(cls, v: str):
        if not str(v).strip():
            raise ValueError("po_no cannot be blank")
        return v

class NLDeletePOParseResult(BaseModel):
    op: Literal["delete_po"] = "delete_po"
    fields: DeletePOSchema

# LLM-structured parsers
structured_create_parser = llm.with_structured_output(NLParseResult)
structured_read_parser = llm.with_structured_output(NLReadParseResult)
structured_update_parser = llm.with_structured_output(NLUpdateParseResult)
structured_delete_invoice_parser = llm.with_structured_output(NLDeleteInvoiceParseResult)
structured_delete_po_parser = llm.with_structured_output(NLDeletePOParseResult)

# =========================
# LangGraph State
# =========================
class AppState(TypedDict):
    # Router selector
    task: Literal[
        "rag_qa",
        "invoice_create",
        "invoice_read",
        "invoice_update",
        "invoice_delete",
        "po_delete",
    ]
    # User prompt
    user_prompt: str

    # --- RAG (stub) ---
    rag_answer: Optional[str]

    # --- Create fields ---
    parsed: Optional[NLParseResult]
    validation_errors: Annotated[list[str], operator.add]
    po_exists: Optional[bool]
    duplicate_invoice: Optional[bool]
    summary_for_approval: Optional[str]
    human_approved: Optional[bool]
    write_ok: Optional[bool]
    write_reason: Optional[str]
    rejection_email: Optional[EmailDraft]

    # --- Read fields ---
    read_parsed: Optional[NLReadParseResult]
    read_results: Optional[List[Dict[str, str]]]
    read_error: Optional[str]

    # --- Update fields ---
    upd_parsed: Optional[NLUpdateParseResult]
    upd_errors: Annotated[list[str], operator.add]
    upd_target_exists: Optional[bool]
    upd_po_exists: Optional[bool]
    upd_summary_for_approval: Optional[str]
    upd_write_ok: Optional[bool]
    upd_write_reason: Optional[str]
    upd_rejection_email: Optional[EmailDraft]
    upd_message: Optional[str]

    # --- Delete Invoice fields ---
    del_inv_parsed: Optional[NLDeleteInvoiceParseResult]
    del_inv_errors: Annotated[list[str], operator.add]
    del_inv_exists: Optional[bool]
    del_inv_summary_for_approval: Optional[str]
    del_inv_write_ok: Optional[bool]
    del_inv_write_reason: Optional[str]

    # --- Delete PO fields ---
    del_po_parsed: Optional[NLDeletePOParseResult]
    del_po_errors: Annotated[list[str], operator.add]
    del_po_exists: Optional[bool]
    del_po_has_refs: Optional[bool]
    del_po_ref_invoices: Optional[List[str]]
    del_po_summary_for_approval: Optional[str]
    del_po_write_ok: Optional[bool]
    del_po_write_reason: Optional[str]

# =========================
# CSV Helpers
# =========================
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def append_csv_row(path: str, fieldnames: List[str], row: dict):
    file_exists = os.path.exists(path)
    write_header = (not file_exists) or (os.path.getsize(path) == 0)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def invoice_exists(invoice_no: str) -> bool:
    rows = read_csv_rows(INVOICE_CSV)
    return any((r.get("invoice_no") or "").strip().lower() == invoice_no.strip().lower() for r in rows)

def po_present(po_no: str) -> bool:
    rows = read_csv_rows(PO_CSV)
    return any((r.get("po_no") or "").strip().lower() == po_no.strip().lower() for r in rows)

def invoices_referencing_po(po_no: str) -> List[str]:
    rows = read_csv_rows(INVOICE_CSV)
    ref = []
    for r in rows:
        if (r.get("po_no") or "").strip().lower() == po_no.strip().lower():
            ref.append(r.get("invoice_no") or "")
    return ref

# =========================
# Node Implementations
# =========================
# --- RAG stub (Chain 1) ---
def rag_qa_node(state: AppState):
    q = state["user_prompt"]
    system = (
        "You are a finance policy assistant. Answer only from the provided context. "
        "If not in context, say you don't know."
    )
    # Plug real retriever later
    context = "CONTEXT_PLACEHOLDER: (Load PDF → chunk → embed → retrieve top-k)."
    prompt = f"{system}\n\nContext:\n{context}\n\nUser question:\n{q}"
    answer = llm.invoke(prompt).content
    return {"rag_answer": answer}

# --- CREATE path nodes (Chain 2a) ---
def parse_create_from_nl(state: AppState):
    user_text = state["user_prompt"]
    parsed = structured_create_parser.invoke(f"User wants to create an invoice.\nText: {user_text}")
    return {"parsed": parsed, "validation_errors": []}

def validate_invoice(state: AppState):
    errors = []
    duplicate = invoice_exists(state["parsed"].fields.invoice_no)
    if duplicate:
        errors.append(f"Duplicate invoice number: {state['parsed'].fields.invoice_no}")
    po_ok = po_present(state["parsed"].fields.po_no)
    if not po_ok:
        errors.append(f"PO not found: {state['parsed'].fields.po_no}")
    return {"validation_errors": errors, "duplicate_invoice": duplicate, "po_exists": po_ok}

def maybe_email_on_po_missing(state: AppState):
    if state["po_exists"]:
        return {}
    f = state["parsed"].fields
    prompt = (
        "Draft a short, professional email informing the submitter that their invoice "
        "cannot be created because the PO was not found. Include bullet points for the details. "
        f"Details:\n- PO Number: {f.po_no}\n- Invoice: {f.invoice_no}\n"
        f"- Customer: {f.customer}\n- Amount: {f.amount}\n- Due date: {f.due_date}"
    )
    email_text = llm.invoke(prompt).content
    lines = [l for l in email_text.splitlines() if l.strip()]
    subject = lines[0][:120] if lines else "Invoice Creation Error: PO Not Found"
    body = "\n".join(lines[1:]) if len(lines) > 1 else email_text
    email = EmailDraft(subject=subject, body=body)
    return {"rejection_email": email}

def build_human_approval_summary(state: AppState):
    f = state["parsed"].fields
    summary = (
        "Please review this invoice creation request:\n"
        f"- Invoice No: {f.invoice_no}\n"
        f"- PO No: {f.po_no}\n"
        f"- Customer: {f.customer}\n"
        f"- Amount: {f.amount}\n"
        f"- Due Date: {f.due_date}\n\n"
        "Reply with APPROVE or REJECT."
    )
    return {"summary_for_approval": summary}

def write_invoice_if_approved(state: AppState):
    if not state.get("human_approved"):
        return {"write_ok": False, "write_reason": "Not approved by human yet."}
    f = state["parsed"].fields
    if invoice_exists(f.invoice_no):
        return {"write_ok": False, "write_reason": "Duplicate invoice at write time."}
    if not po_present(f.po_no):
        return {"write_ok": False, "write_reason": "PO missing at write time."}
    append_csv_row(
        INVOICE_CSV,
        fieldnames=["invoice_no", "po_no", "customer", "amount", "due_date"],
        row={
            "invoice_no": f.invoice_no,
            "po_no": f.po_no,
            "customer": f.customer,
            "amount": f.amount,
            "due_date": f.due_date,
        },
    )
    return {"write_ok": True, "write_reason": "Written"}

# --- READ path nodes (Chain 2b) ---
def parse_read_from_nl(state: AppState):
    user_text = state["user_prompt"]
    parsed = structured_read_parser.invoke(
        "User wants to read invoices. Extract as many filters as possible from the text.\n"
        f"Text: {user_text}"
    )
    return {"read_parsed": parsed, "read_error": None}

def perform_read(state: AppState):
    """Filter invoices.csv using only known columns; ranges apply to due_date (inclusive)."""
    try:
        rows = read_csv_rows(INVOICE_CSV)
        if not rows:
            return {"read_results": [], "read_error": None}

        f = state["read_parsed"].filters

        def norm(s: Optional[str]) -> str:
            return (s or "").strip().lower()

        def to_date(s: Optional[str]):
            if not s:
                return None
            return datetime.strptime(s, "%Y-%m-%d").date()

        due_from = to_date(f.due_from)
        due_to = to_date(f.due_to)

        results: List[Dict[str, str]] = []
        for r in rows:
            ok = True
            if f.invoice_no and norm(r.get("invoice_no")) != norm(f.invoice_no):
                ok = False
            if ok and f.po_no and norm(r.get("po_no")) != norm(f.po_no):
                ok = False
            if ok and f.customer and norm(f.customer) not in norm(r.get("customer")):
                ok = False
            if ok and (due_from or due_to):
                try:
                    d = datetime.strptime(r.get("due_date", ""), "%Y-%m-%d").date()
                except Exception:
                    ok = False
                if ok and due_from and d < due_from:
                    ok = False
                if ok and due_to and d > due_to:
                    ok = False
            if ok:
                results.append(r)

        return {"read_results": results, "read_error": None}
    except Exception as e:
        return {"read_results": [], "read_error": str(e)}

# --- UPDATE path nodes (Chain 2c) ---
def parse_update_from_nl(state: AppState):
    user_text = state["user_prompt"]
    parsed = structured_update_parser.invoke(
        "User wants to update an existing invoice. Extract target invoice_no and any fields to change.\n"
        "Do NOT change invoice_no itself; only set new_* fields when present.\n"
        f"Text: {user_text}"
    )
    return {"upd_parsed": parsed, "upd_errors": []}

def validate_update(state: AppState):
    errors: List[str] = []
    target = state["upd_parsed"].fields.invoice_no

    target_exists = invoice_exists(target)
    if not target_exists:
        errors.append(f"Invoice not found: {target}")

    po_ok = True
    new_po = state["upd_parsed"].fields.new_po_no
    if new_po:
        po_ok = po_present(new_po)
        if not po_ok:
            errors.append(f"PO not found: {new_po}")

    return {
        "upd_errors": errors,
        "upd_target_exists": target_exists,
        "upd_po_exists": po_ok,
    }

def maybe_email_on_update_po_missing(state: AppState):
    upd = state["upd_parsed"].fields
    if (upd.new_po_no is None) or state["upd_po_exists"]:
        return {}
    prompt = (
        "Draft a short, professional email informing the submitter that their invoice update "
        "cannot proceed because the new PO was not found. Include bullet points for details. "
        f"Details:\n- Target Invoice: {upd.invoice_no}\n"
        f"- Requested New PO: {upd.new_po_no}\n"
        f"- Other changes: customer={upd.new_customer}, amount={upd.new_amount}, due_date={upd.new_due_date}"
    )
    email_text = llm.invoke(prompt).content
    lines = [l for l in email_text.splitlines() if l.strip()]
    subject = lines[0][:120] if lines else "Invoice Update Error: PO Not Found"
    body = "\n".join(lines[1:]) if len(lines) > 1 else email_text
    email = EmailDraft(subject=subject, body=body)
    return {"upd_rejection_email": email, "upd_message": f"PO not found: {upd.new_po_no}"}

def build_update_approval_summary(state: AppState):
    f = state["upd_parsed"].fields
    parts = []
    if f.new_po_no is not None:
        parts.append(f"PO No: {f.new_po_no}")
    if f.new_customer is not None:
        parts.append(f"Customer: {f.new_customer}")
    if f.new_amount is not None:
        parts.append(f"Amount: {f.new_amount}")
    if f.new_due_date is not None:
        parts.append(f"Due Date: {f.new_due_date}")

    if not parts:
        summary = (
            f"No changes detected for invoice {f.invoice_no}. "
            "Provide at least one field to update (PO, customer, amount, or due date)."
        )
    else:
        summary = (
            f"Please review this invoice update request for {f.invoice_no}:\n"
            "- Changes:\n  - " + "\n  - ".join(parts) + "\n\nReply with APPROVE or REJECT."
        )
    return {"upd_summary_for_approval": summary}

def write_update_if_approved(state: AppState):
    if not state.get("human_approved"):
        return {"upd_write_ok": False, "upd_write_reason": "Not approved by human yet."}

    f = state["upd_parsed"].fields
    rows = read_csv_rows(INVOICE_CSV)
    if not rows:
        return {"upd_write_ok": False, "upd_write_reason": "No invoices file or empty."}

    idx = None
    for i, r in enumerate(rows):
        if (r.get("invoice_no") or "").strip().lower() == f.invoice_no.strip().lower():
            idx = i
            break
    if idx is None:
        return {"upd_write_ok": False, "upd_write_reason": f"Invoice {f.invoice_no} not found at write time."}

    if f.new_po_no is not None and not po_present(f.new_po_no):
        return {"upd_write_ok": False, "upd_write_reason": "PO missing at write time."}

    new_row = dict(rows[idx])
    if f.new_po_no is not None:
        new_row["po_no"] = f.new_po_no
    if f.new_customer is not None:
        new_row["customer"] = f.new_customer
    if f.new_amount is not None:
        new_row["amount"] = str(f.new_amount)
    if f.new_due_date is not None:
        new_row["due_date"] = f.new_due_date

    rows[idx] = new_row
    write_csv_rows(INVOICE_CSV, ["invoice_no", "po_no", "customer", "amount", "due_date"], rows)
    return {"upd_write_ok": True, "upd_write_reason": "Updated"}

# --- DELETE INVOICE path nodes (Chain 2d) ---
def parse_delete_invoice_from_nl(state: AppState):
    user_text = state["user_prompt"]
    parsed = structured_delete_invoice_parser.invoke(
        "User wants to delete an invoice. Extract the invoice_no to delete.\n"
        f"Text: {user_text}"
    )
    return {"del_inv_parsed": parsed, "del_inv_errors": []}

def validate_delete_invoice(state: AppState):
    inv = state["del_inv_parsed"].fields.invoice_no
    exists = invoice_exists(inv)
    errors: List[str] = []
    if not exists:
        errors.append(f"Invoice not found: {inv}")
    return {"del_inv_errors": errors, "del_inv_exists": exists}

def build_delete_invoice_approval_summary(state: AppState):
    inv = state["del_inv_parsed"].fields.invoice_no
    summary = (
        f"Please review this invoice deletion request:\n"
        f"- Invoice No: {inv}\n\n"
        "Reply with APPROVE or REJECT."
    )
    return {"del_inv_summary_for_approval": summary}

def write_delete_invoice_if_approved(state: AppState):
    if not state.get("human_approved"):
        return {"del_inv_write_ok": False, "del_inv_write_reason": "Not approved by human yet."}
    inv = state["del_inv_parsed"].fields.invoice_no
    rows = read_csv_rows(INVOICE_CSV)
    if not rows:
        return {"del_inv_write_ok": False, "del_inv_write_reason": "Invoices file empty or missing."}
    new_rows = [r for r in rows if (r.get("invoice_no") or "").strip().lower() != inv.strip().lower()]
    if len(new_rows) == len(rows):
        return {"del_inv_write_ok": False, "del_inv_write_reason": f"Invoice {inv} not found at write time."}
    write_csv_rows(INVOICE_CSV, ["invoice_no", "po_no", "customer", "amount", "due_date"], new_rows)
    return {"del_inv_write_ok": True, "del_inv_write_reason": "Invoice deleted"}

# --- DELETE PO path nodes (Chain 2e) ---
def parse_delete_po_from_nl(state: AppState):
    user_text = state["user_prompt"]
    parsed = structured_delete_po_parser.invoke(
        "User wants to delete a purchase order. Extract the po_no to delete.\n"
        f"Text: {user_text}"
    )
    return {"del_po_parsed": parsed, "del_po_errors": []}

def validate_delete_po(state: AppState):
    po = state["del_po_parsed"].fields.po_no
    exists = po_present(po)
    errors: List[str] = []
    refs = invoices_referencing_po(po)
    has_refs = len(refs) > 0

    if not exists:
        errors.append(f"PO not found: {po}")
    if has_refs:
        errors.append(f"PO {po} is referenced by {len(refs)} invoice(s): {', '.join(refs)}")

    return {
        "del_po_errors": errors,
        "del_po_exists": exists,
        "del_po_has_refs": has_refs,
        "del_po_ref_invoices": refs,
    }

def build_delete_po_approval_summary(state: AppState):
    po = state["del_po_parsed"].fields.po_no
    refs = state.get("del_po_ref_invoices") or []
    if refs:
        note = f"\nWARNING: This PO is referenced by {len(refs)} invoice(s): {', '.join(refs)}"
    else:
        note = ""
    summary = (
        f"Please review this PO deletion request:\n"
        f"- PO No: {po}{note}\n\n"
        "Reply with APPROVE or REJECT."
    )
    return {"del_po_summary_for_approval": summary}

def write_delete_po_if_approved(state: AppState):
    if not state.get("human_approved"):
        return {"del_po_write_ok": False, "del_po_write_reason": "Not approved by human yet."}

    po = state["del_po_parsed"].fields.po_no
    # Guard: do not allow deletion if referenced by invoices
    refs = invoices_referencing_po(po)
    if refs:
        return {"del_po_write_ok": False, "del_po_write_reason": f"PO is referenced by invoices: {', '.join(refs)}"}

    rows = read_csv_rows(PO_CSV)
    if not rows:
        return {"del_po_write_ok": False, "del_po_write_reason": "PO file empty or missing."}
    new_rows = [r for r in rows if (r.get("po_no") or "").strip().lower() != po.strip().lower()]
    if len(new_rows) == len(rows):
        return {"del_po_write_ok": False, "del_po_write_reason": f"PO {po} not found at write time."}
    write_csv_rows(PO_CSV, ["po_no", "supplier", "amount_hint"], new_rows)
    return {"del_po_write_ok": True, "del_po_write_reason": "PO deleted"}

# =========================
# Graph Wiring
# =========================
graph = StateGraph(AppState)

# Common nodes
graph.add_node("rag_qa", rag_qa_node)

# CREATE nodes
graph.add_node("parse_create", parse_create_from_nl)
graph.add_node("validate", validate_invoice)
graph.add_node("email_if_po_missing", maybe_email_on_po_missing)
graph.add_node("approval_summary", build_human_approval_summary)
graph.add_node("write_if_approved", write_invoice_if_approved)

# READ nodes
graph.add_node("parse_read", parse_read_from_nl)
graph.add_node("read_execute", perform_read)

# UPDATE nodes
graph.add_node("parse_update", parse_update_from_nl)
graph.add_node("validate_update", validate_update)
graph.add_node("email_if_update_po_missing", maybe_email_on_update_po_missing)
graph.add_node("update_approval_summary", build_update_approval_summary)
graph.add_node("write_update_if_approved", write_update_if_approved)

# DELETE INVOICE nodes
graph.add_node("parse_delete_invoice", parse_delete_invoice_from_nl)
graph.add_node("validate_delete_invoice", validate_delete_invoice)
graph.add_node("delete_invoice_approval_summary", build_delete_invoice_approval_summary)
graph.add_node("write_delete_invoice_if_approved", write_delete_invoice_if_approved)

# DELETE PO nodes
graph.add_node("parse_delete_po", parse_delete_po_from_nl)
graph.add_node("validate_delete_po", validate_delete_po)
graph.add_node("delete_po_approval_summary", build_delete_po_approval_summary)
graph.add_node("write_delete_po_if_approved", write_delete_po_if_approved)

# Router
def start_router(state: AppState):
    if state["task"] == "rag_qa":
        return "rag"
    elif state["task"] == "invoice_read":
        return "read"
    elif state["task"] == "invoice_update":
        return "update"
    elif state["task"] == "invoice_delete":
        return "del_inv"
    elif state["task"] == "po_delete":
        return "del_po"
    else:
        return "create"

graph.add_conditional_edges(
    START,
    start_router,
    {
        "rag": "rag_qa",
        "create": "parse_create",
        "read": "parse_read",
        "update": "parse_update",
        "del_inv": "parse_delete_invoice",
        "del_po": "parse_delete_po",
    },
)

# RAG path
graph.add_edge("rag_qa", END)

# CREATE path
graph.add_edge("parse_create", "validate")

def has_missing_po_create(state: AppState):
    return not state["po_exists"]

graph.add_conditional_edges("validate", has_missing_po_create, {
    True: "email_if_po_missing",
    False: "approval_summary",
})

graph.add_edge("email_if_po_missing", END)
graph.add_edge("approval_summary", "write_if_approved")
graph.add_edge("write_if_approved", END)

# READ path
graph.add_edge("parse_read", "read_execute")
graph.add_edge("read_execute", END)

# UPDATE path
graph.add_edge("parse_update", "validate_update")

def has_missing_po_update(state: AppState):
    upd = state["upd_parsed"].fields
    return (upd.new_po_no is not None) and (not state["upd_po_exists"])

graph.add_conditional_edges("validate_update", has_missing_po_update, {
    True: "email_if_update_po_missing",
    False: "update_approval_summary",
})

graph.add_edge("email_if_update_po_missing", END)
graph.add_edge("update_approval_summary", "write_update_if_approved")
graph.add_edge("write_update_if_approved", END)

# DELETE INVOICE path
graph.add_edge("parse_delete_invoice", "validate_delete_invoice")
graph.add_edge("validate_delete_invoice", "delete_invoice_approval_summary")
graph.add_edge("delete_invoice_approval_summary", "write_delete_invoice_if_approved")
graph.add_edge("write_delete_invoice_if_approved", END)

# DELETE PO path
graph.add_edge("parse_delete_po", "validate_delete_po")
graph.add_edge("validate_delete_po", "delete_po_approval_summary")
graph.add_edge("delete_po_approval_summary", "write_delete_po_if_approved")
graph.add_edge("write_delete_po_if_approved", END)

# Compile
app = graph.compile()

# =========================
# Optional CLI (manual test)
# =========================
if __name__ == "__main__":
    print("Mode options: create / read / update / delete-invoice / delete-po / rag (exit to quit)")
    while True:
        mode = input("\nMode: ").strip().lower()
        if mode in ("exit", "quit", "q"):
            break
        prompt = input("Your prompt: ").strip()

        if mode == "rag":
            out = app.invoke({"task": "rag_qa", "user_prompt": prompt})
            print("RAG answer:", out.get("rag_answer"))

        elif mode == "read":
            out = app.invoke({"task": "invoice_read", "user_prompt": prompt})
            print("Read error:", out.get("read_error"))
            print("Results:", out.get("read_results"))

        elif mode == "create":
            out = app.invoke({"task": "invoice_create", "user_prompt": prompt, "human_approved": False})
            if out.get("rejection_email"):
                print("\n--- PO Missing Email Draft ---")
                print("SUBJECT:", out["rejection_email"].subject)
                print("BODY:\n", out["rejection_email"].body)
                continue
            errors = out.get("validation_errors") or []
            if errors:
                print("\nValidation Errors:")
                for e in errors:
                    print("-", e)
                continue
            if out.get("summary_for_approval"):
                print("\n--- Approval Summary ---")
                print(out["summary_for_approval"])
                ok = input("Approve? (y/n): ").strip().lower() in ("y", "yes")
                out2 = app.invoke({"task": "invoice_create", "user_prompt": prompt, "human_approved": ok})
                print("Write OK:", out2.get("write_ok"), "| Reason:", out2.get("write_reason"))

        elif mode == "update":
            out = app.invoke({"task": "invoice_update", "user_prompt": prompt, "human_approved": False})
            if out.get("upd_rejection_email") or out.get("upd_message"):
                if out.get("upd_rejection_email"):
                    em = out["upd_rejection_email"]
                    print("\n--- Update PO Missing Email ---")
                    print("SUBJECT:", em.subject)
                    print("BODY:\n", em.body)
                if out.get("upd_message"):
                    print("Message:", out["upd_message"])
                continue
            errors = out.get("upd_errors") or []
            if errors:
                print("\nUpdate Errors:")
                for e in errors:
                    print("-", e)
                continue
            if out.get("upd_summary_for_approval"):
                print("\n--- Update Approval Summary ---")
                print(out["upd_summary_for_approval"])
                ok = input("Approve update? (y/n): ").strip().lower() in ("y", "yes")
                out2 = app.invoke({"task": "invoice_update", "user_prompt": prompt, "human_approved": ok})
                print("Update OK:", out2.get("upd_write_ok"), "| Reason:", out2.get("upd_write_reason"))

        elif mode == "delete-invoice":
            out = app.invoke({"task": "invoice_delete", "user_prompt": prompt, "human_approved": False})
            errors = out.get("del_inv_errors") or []
            if errors:
                print("\nDelete Invoice Errors:")
                for e in errors:
                    print("-", e)
                continue
            if out.get("del_inv_summary_for_approval"):
                print("\n--- Delete Invoice Approval Summary ---")
                print(out["del_inv_summary_for_approval"])
                ok = input("Approve delete? (y/n): ").strip().lower() in ("y", "yes")
                out2 = app.invoke({"task": "invoice_delete", "user_prompt": prompt, "human_approved": ok})
                print("Delete Invoice OK:", out2.get("del_inv_write_ok"), "| Reason:", out2.get("del_inv_write_reason"))

        elif mode == "delete-po":
            out = app.invoke({"task": "po_delete", "user_prompt": prompt, "human_approved": False})
            errors = out.get("del_po_errors") or []
            if errors:
                print("\nDelete PO Errors:")
                for e in errors:
                    print("-", e)
                continue
            if out.get("del_po_summary_for_approval"):
                print("\n--- Delete PO Approval Summary ---")
                print(out["del_po_summary_for_approval"])
                ok = input("Approve delete? (y/n): ").strip().lower() in ("y", "yes")
                out2 = app.invoke({"task": "po_delete", "user_prompt": prompt, "human_approved": ok})
                print("Delete PO OK:", out2.get("del_po_write_ok"), "| Reason:", out2.get("del_po_write_reason"))
        else:
            print("Unknown mode.")
