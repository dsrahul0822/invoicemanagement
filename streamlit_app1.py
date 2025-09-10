# streamlit_app.py
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from datetime import date

# --- Import your compiled graph + CSV paths from backend ---
from app1 import app, INVOICE_CSV, PO_CSV

# =========================
# Helpers
# =========================
def fmt_date(d: date | None) -> str | None:
    return d.strftime("%Y-%m-%d") if d else None

def run_flow(task: str, user_prompt: str, human_approved: bool = False) -> dict:
    state = {"task": task, "user_prompt": user_prompt}
    # For write steps, add approval flag
    if task in ("invoice_create", "invoice_update", "invoice_delete", "po_delete"):
        state["human_approved"] = human_approved
    return app.invoke(state)

def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        return pd.DataFrame({"error": [f"Failed to read {path}: {e}"]})

def email_to_subject_body(email_obj_or_dict):
    # Works both for Pydantic model and dict
    if not email_obj_or_dict:
        return "", ""
    try:
        subject = email_obj_or_dict.subject
        body = email_obj_or_dict.body
    except Exception:
        subject = getattr(email_obj_or_dict, "subject", "") or (email_obj_or_dict.get("subject") if isinstance(email_obj_or_dict, dict) else "")
        body = getattr(email_obj_or_dict, "body", "") or (email_obj_or_dict.get("body") if isinstance(email_obj_or_dict, dict) else "")
    return subject, body

# --- Build NL prompts that your backend parsers understand cleanly ---
def make_create_prompt(inv_no, po_no, customer, amount, due_date):
    return f"Create an invoice {inv_no} for {customer} for {amount} due {due_date} with {po_no}"

def make_read_prompt(invoice_no, po_no, customer, due_from, due_to):
    parts = []
    if invoice_no: parts.append(f"with invoice {invoice_no}")
    if po_no: parts.append(f"with PO {po_no}")
    if customer: parts.append(f"for {customer}")
    if due_from and due_to: parts.append(f"due between {due_from} and {due_to}")
    elif due_from: parts.append(f"due on or after {due_from}")
    elif due_to: parts.append(f"due on or before {due_to}")
    suffix = " ".join(parts) if parts else "all invoices"
    return f"Show invoices {suffix}."

def make_update_prompt(invoice_no, new_po_no, new_customer, new_amount, new_due_date):
    changes = []
    if new_po_no: changes.append(f"PO to {new_po_no}")
    if new_customer: changes.append(f"customer to {new_customer}")
    if new_amount is not None: changes.append(f"amount to {new_amount}")
    if new_due_date: changes.append(f"due date to {new_due_date}")
    if not changes:
        return f"Update invoice {invoice_no}: no changes provided"
    return f"Update invoice {invoice_no}: set " + "; ".join(changes)

def make_delete_invoice_prompt(invoice_no):
    return f"Delete invoice {invoice_no}"

def make_delete_po_prompt(po_no):
    return f"Delete PO {po_no}"

def approval_ready(task_key: str, out: dict) -> bool:
    if task_key == "invoice_create":
        return bool(out.get("summary_for_approval"))
    if task_key == "invoice_update":
        return bool(out.get("upd_summary_for_approval"))
    if task_key == "invoice_delete":
        return bool(out.get("del_inv_summary_for_approval"))
    if task_key == "po_delete":
        return bool(out.get("del_po_summary_for_approval"))
    return False

# =========================
# Page
# =========================
st.set_page_config(page_title="Invoice System (LangGraph)", layout="centered")
st.title("🧾 Invoice System — Clean UI")

# Init session keys
for k, v in {
    "last_task": None,
    "last_prompt": None,
    "last_out": None,
    "ready_for_approval": False,
}.items():
    st.session_state.setdefault(k, v)

# --- Mode switch ---
mode = st.radio(
    "Choose an action",
    options=["RAG Q&A", "Create", "Read", "Update", "Delete"],
    horizontal=True,
)
task_key = {
    "RAG Q&A": "rag_qa",
    "Create": "invoice_create",
    "Read": "invoice_read",
    "Update": "invoice_update",
    "Delete": "delete",  # special, we choose sub-type below
}[mode]

st.divider()

# =========================
# Forms per mode
# =========================

# --- RAG ---
if task_key == "rag_qa":
    q = st.text_input("Ask a question (RAG)", placeholder="e.g., What is the late payment penalty?")
    col = st.columns(2)[0]
    if col.button("Ask"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            out = run_flow("rag_qa", q, human_approved=False)
            st.session_state.update({"last_task": "rag_qa", "last_prompt": q, "last_out": out, "ready_for_approval": False})
    if st.session_state.get("last_task") == "rag_qa" and st.session_state.get("last_out"):
        st.success("Answer")
        st.write(st.session_state["last_out"].get("rag_answer", "No answer."))

# --- Create (Invoice) ---
elif task_key == "invoice_create":
    with st.form("create_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            inv_no = st.text_input("Invoice No*", placeholder="INV-0210")
            po_no = st.text_input("PO No*", placeholder="PO-1003")
            amount = st.number_input("Amount*", min_value=0.0, step=1.0, format="%.2f")
        with c2:
            customer = st.text_input("Customer*", placeholder="AcmeCorp")
            due = st.date_input("Due Date*", value=None)
        submitted = st.form_submit_button("Validate")
    if submitted:
        if not (inv_no and po_no and customer and due):
            st.warning("Please fill all required fields.")
        else:
            prompt = make_create_prompt(inv_no, po_no, customer, amount, fmt_date(due))
            out = run_flow("invoice_create", prompt, human_approved=False)
            st.session_state.update({"last_task": "invoice_create", "last_prompt": prompt, "last_out": out})
            st.session_state["ready_for_approval"] = approval_ready("invoice_create", out)

    # Render outcome
    out = st.session_state.get("last_out") if st.session_state.get("last_task") == "invoice_create" else None
    if out:
        errs = out.get("validation_errors") or []
        if errs:
            st.error("Validation Errors")
            for e in errs: st.write(f"- {e}")

        email = out.get("rejection_email")
        if email:
            st.warning("PO Missing — copy this email")
            subj, body = email_to_subject_body(email)
            st.markdown(f"**Subject:** {subj}")
            st.text_area("Body", value=body, height=180)

        if out.get("summary_for_approval"):
            st.info("Approval Summary")
            st.text_area("Summary", value=out["summary_for_approval"], height=140)

        if st.session_state.get("ready_for_approval"):
            if st.button("✅ Approve & Write"):
                out2 = run_flow("invoice_create", st.session_state["last_prompt"], human_approved=True)
                st.session_state["last_out"] = out2
                st.session_state["ready_for_approval"] = False

        if out.get("write_ok") is not None:
            if out.get("write_ok"): st.success(f"Invoice written ({out.get('write_reason')})")
            else: st.info(f"Not written: {out.get('write_reason')}")

# --- Read (Invoices) ---
elif task_key == "invoice_read":
    with st.form("read_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            r_inv = st.text_input("Invoice No", placeholder="INV-0003")
            r_po = st.text_input("PO No", placeholder="PO-1002")
            r_customer = st.text_input("Customer", placeholder="AcmeCorp")
        with c2:
            r_from = st.date_input("Due From", value=None)
            r_to = st.date_input("Due To", value=None)
        search = st.form_submit_button("Search")
    if search:
        prompt = make_read_prompt(r_inv, r_po, r_customer, fmt_date(r_from), fmt_date(r_to))
        out = run_flow("invoice_read", prompt, human_approved=False)
        st.session_state.update({"last_task": "invoice_read", "last_prompt": prompt, "last_out": out, "ready_for_approval": False})

    out = st.session_state.get("last_out") if st.session_state.get("last_task") == "invoice_read" else None
    if out:
        if out.get("read_error"): st.error(out["read_error"])
        df = pd.DataFrame(out.get("read_results") or [])
        if df.empty:
            st.info("No invoices matched your filters.")
        else:
            st.success(f"Found {len(df)} invoice(s).")
            st.dataframe(df, use_container_width=True)

# --- Update (Invoice) ---
elif task_key == "invoice_update":
    with st.form("update_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            u_inv = st.text_input("Target Invoice No*", placeholder="INV-0210")
            u_po = st.text_input("New PO No", placeholder="PO-2001")
            u_amount_str = st.text_input("New Amount", placeholder="e.g., 6000")
        with c2:
            u_customer = st.text_input("New Customer", placeholder="AcmeCorp India")
            u_due = st.date_input("New Due Date", value=None)
        submitted = st.form_submit_button("Validate")
    if submitted:
        if not u_inv:
            st.warning("Please provide the target invoice number.")
        else:
            try:
                u_amount = float(u_amount_str) if u_amount_str.strip() else None
            except Exception:
                st.warning("Amount must be a number (e.g., 6000).")
                u_amount = None
            prompt = make_update_prompt(
                u_inv,
                u_po.strip() or None,
                u_customer.strip() or None,
                u_amount,
                fmt_date(u_due),
            )
            out = run_flow("invoice_update", prompt, human_approved=False)
            st.session_state.update({"last_task": "invoice_update", "last_prompt": prompt, "last_out": out})
            st.session_state["ready_for_approval"] = approval_ready("invoice_update", out)

    out = st.session_state.get("last_out") if st.session_state.get("last_task") == "invoice_update" else None
    if out:
        # PO missing path (email/message)
        if out.get("upd_message"): st.error(out["upd_message"])
        if out.get("upd_rejection_email"):
            st.warning("PO Missing — copy this email")
            subj, body = email_to_subject_body(out["upd_rejection_email"])
            st.markdown(f"**Subject:** {subj}")
            st.text_area("Body", value=body, height=180)

        errs = out.get("upd_errors") or []
        if errs:
            st.error("Update Errors")
            for e in errs: st.write(f"- {e}")

        if out.get("upd_summary_for_approval"):
            st.info("Update Approval Summary")
            st.text_area("Summary", value=out["upd_summary_for_approval"], height=140)

        if st.session_state.get("ready_for_approval"):
            if st.button("✅ Approve & Write"):
                out2 = run_flow("invoice_update", st.session_state["last_prompt"], human_approved=True)
                st.session_state["last_out"] = out2
                st.session_state["ready_for_approval"] = False

        if out.get("upd_write_ok") is not None:
            if out.get("upd_write_ok"): st.success(f"Invoice updated ({out.get('upd_write_reason')})")
            else: st.info(f"Not updated: {out.get('upd_write_reason')}")

# --- Delete (Invoice or PO) ---
else:  # task_key == "delete"
    sub = st.radio("What do you want to delete?", options=["Invoice", "PO"], horizontal=True)
    if sub == "Invoice":
        with st.form("del_inv_form", clear_on_submit=False):
            d_inv = st.text_input("Invoice No*", placeholder="INV-0210")
            go = st.form_submit_button("Validate")
        if go:
            if not d_inv:
                st.warning("Please provide an invoice number.")
            else:
                prompt = make_delete_invoice_prompt(d_inv)
                out = run_flow("invoice_delete", prompt, human_approved=False)
                st.session_state.update({"last_task": "invoice_delete", "last_prompt": prompt, "last_out": out})
                st.session_state["ready_for_approval"] = approval_ready("invoice_delete", out)

        out = st.session_state.get("last_out") if st.session_state.get("last_task") == "invoice_delete" else None
        if out:
            errs = out.get("del_inv_errors") or []
            if errs:
                st.error("Delete Invoice Errors")
                for e in errs: st.write(f"- {e}")

            if out.get("del_inv_summary_for_approval"):
                st.info("Delete Invoice Approval Summary")
                st.text_area("Summary", value=out["del_inv_summary_for_approval"], height=120)

            if st.session_state.get("ready_for_approval"):
                if st.button("✅ Approve & Delete"):
                    out2 = run_flow("invoice_delete", st.session_state["last_prompt"], human_approved=True)
                    st.session_state["last_out"] = out2
                    st.session_state["ready_for_approval"] = False

            if out.get("del_inv_write_ok") is not None:
                if out.get("del_inv_write_ok"): st.success(f"Invoice deleted ({out.get('del_inv_write_reason')})")
                else: st.info(f"Not deleted: {out.get('del_inv_write_reason')}")

    else:  # Delete PO
        with st.form("del_po_form", clear_on_submit=False):
            d_po = st.text_input("PO No*", placeholder="PO-2001")
            go = st.form_submit_button("Validate")
        if go:
            if not d_po:
                st.warning("Please provide a PO number.")
            else:
                prompt = make_delete_po_prompt(d_po)
                out = run_flow("po_delete", prompt, human_approved=False)
                st.session_state.update({"last_task": "po_delete", "last_prompt": prompt, "last_out": out})
                st.session_state["ready_for_approval"] = approval_ready("po_delete", out)

        out = st.session_state.get("last_out") if st.session_state.get("last_task") == "po_delete" else None
        if out:
            errs = out.get("del_po_errors") or []
            if errs:
                st.error("Delete PO Errors")
                for e in errs: st.write(f"- {e}")

            if out.get("del_po_summary_for_approval"):
                st.info("Delete PO Approval Summary")
                st.text_area("Summary", value=out["del_po_summary_for_approval"], height=120)

            if st.session_state.get("ready_for_approval"):
                if st.button("✅ Approve & Delete"):
                    out2 = run_flow("po_delete", st.session_state["last_prompt"], human_approved=True)
                    st.session_state["last_out"] = out2
                    st.session_state["ready_for_approval"] = False

            if out.get("del_po_write_ok") is not None:
                if out.get("del_po_write_ok"): st.success(f"PO deleted ({out.get('del_po_write_reason')})")
                else: st.info(f"Not deleted: {out.get('del_po_write_reason')}")

st.divider()

# --- Minimal data preview (collapsed for cleanliness) ---
with st.expander("Data Preview", expanded=False):
    t1, t2 = st.columns(2)
    with t1:
        st.caption(f"Invoices CSV: {INVOICE_CSV}")
        st.dataframe(read_csv_safe(INVOICE_CSV), use_container_width=True, height=260)
    with t2:
        st.caption(f"PO CSV: {PO_CSV}")
        st.dataframe(read_csv_safe(PO_CSV), use_container_width=True, height=260)
