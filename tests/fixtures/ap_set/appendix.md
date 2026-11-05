# AP-SOP-04 Appendices

## Appendix A. Tools available to the AP Assistant

| Tool | Access |
|---|---|
| `ap.get_invoice`, `ap.list_invoices`, `vendor.get_vendor`, `fx.get_rate`, `approvals.get_approval`, `treasury.get_bulletins` | read |
| `ap.create_invoice`, `ap.match_invoice`, `ap.update_invoice_status` | write |
| `approvals.request_approval` | write |
| `vendor.screen_sanctions` | write (creates a screening record) |
| `ap.create_payment`, `ap.schedule_payment_run` | write, money movement |
| `notify.send_email`, `notify.post_message`, `audit.log_event` | write |

## Appendix B. US holidays (2026)

Jan 1, Jan 19, Feb 16, May 25, Jun 19, Jul 3, Sep 7, Nov 26, Dec 25.

## Appendix C. Hold notice template

> Invoice {invoice_number} from {vendor_name} ({vendor_id}) is on hold. Reason: {hold_reason}. To release it: {action_needed}. Contact: ap@northwind.example.

---
