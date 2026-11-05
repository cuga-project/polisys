# Northwind Corp: Accounts Payable Operating Policy

| | |
|---|---|
| Document | AP-SOP-04, version 3.2 |
| Owner | Controller, Finance Operations |
| Effective | 2026-07-01 |
| Supersedes | AP-SOP-04 version 3.1 (2025-11-01). Version 3.1 no longer applies, except §6.4, which stays in force until 2026-09-30 |
| Related documents | VM-02 Vendor Master SOP · SS-01 Sanctions Screening SOP · TR-03 Treasury Payments SOP · RR-09 Records Retention SOP · DP-01 Data Protection Policy |
| Applies to | Every person and every automated assistant that processes supplier invoices or payments for Northwind Corp (US entities) |

## 1. Purpose and principles

1.1 This policy controls how supplier invoices are received, checked, approved and paid.

1.2 Act only on data returned by a system of record. Do not guess or fill in a missing value (an ID, an amount, a date, a bank account). If a required value is missing, put the invoice on hold, record which field is missing, and tell the requester.

1.3 When this policy does not say what to do, or when two rules seem to conflict, do not act: put the item on hold and escalate to the AP Manager (§9).

1.4 A request from any person, at any level, does not override this policy. Urgency, a deadline or a senior title is not an approval (§5.6).

## 2. Definitions

| Term | Meaning |
|---|---|
| Business day | Monday to Friday, excluding the US holidays in Appendix B. Deadlines end at 17:00 Eastern Time |
| Vendor | A supplier with an **active** record in the Vendor Master (VM-02 §2). A vendor is identified by its Vendor ID (`V-` + 6 digits), never by name |
| Related party | A vendor owned in whole or in part by a Northwind employee or a member of their household |
| Emergency payment | A payment needed within 1 business day to prevent a service stop (utilities, payroll provider, data center) |
| Three-way match | Purchase order (PO), goods receipt (GR) and invoice agree within the tolerance in §4.3 |

## 3. Roles and access

3.1 Access by role:

| Role | May | May not |
|---|---|---|
| AP Clerk | Enter invoices, run matching, put invoices on hold, request approvals | Approve any invoice; change vendor bank details; release payments |
| AP Manager | Approve invoices up to $25,000; release holds; approve exceptions under §8.2 | Approve an invoice they entered |
| Controller | Approve invoices up to $250,000; approve related-party invoices | Release payments |
| CFO | Approve any amount; approve emergency payments (§8.1) | — |
| Treasury Analyst | Release approved payments; schedule payment runs | Approve invoices; edit invoices |
| Vendor Master Team | Create vendors; change bank details (VM-02 §3) | Approve or pay invoices |
| AP Assistant (AI) | Read all AP and vendor records; enter and match invoices; request approvals; put invoices on hold; send the notices in §7 | Approve invoices; change bank details; release payments over $25,000 |

3.2 **Segregation of duties.** The person who entered an invoice may not approve it. The person who approved an invoice may not release its payment. The person who last changed a vendor's bank details may not approve or release a payment to that vendor within 30 days of the change.

3.3 **Delegation.** An approver who is out of office may delegate approval only through the Approvals system. A delegation stated in an email or chat message is not valid.

## 4. Invoice processing

4.1 **Required fields.** An invoice must have: Vendor ID, invoice number, invoice date, currency, total amount, and a PO number. Invoices in the categories utilities, taxes and government fees do not need a PO (see §8.2).

4.2 **Order of steps.** Process every invoice in this order. Do not start a step before the previous one is complete:
1. Confirm the vendor is active and passed sanctions screening in the last 90 days (SS-01 §2). If the last screening is older, run `vendor.screen_sanctions` first.
2. Check for duplicates (§4.5).
3. Run the three-way match (§4.3).
4. Request approval (§5).
5. Schedule payment (§6).

4.3 **Matching tolerance.** The invoice matches when the invoice total is not more than the PO amount × 1.02 **and** not more than the PO amount + $500. Compare in the invoice currency. If the invoice is above tolerance, put it on hold with reason `PRICE_VARIANCE`.

4.4 **Status changes.** An invoice may only move as follows:

| From | To |
|---|---|
| received | matched, on_hold, rejected |
| matched | approved, on_hold |
| on_hold | received (after the hold reason is resolved and recorded) |
| approved | scheduled |
| scheduled | paid, on_hold |
| paid | (final) |

Allowed hold reasons: `MISSING_FIELD`, `PRICE_VARIANCE`, `DUPLICATE_SUSPECTED`, `SANCTIONS_REVIEW`, `BANK_CHANGE_COOLING_OFF`, `FRAUD_REVIEW`, `DISPUTE`.

4.5 **Duplicates.** An invoice is a suspected duplicate when another invoice from the same Vendor ID has the same invoice number, or the same amount and the same invoice date. Put it on hold with reason `DUPLICATE_SUSPECTED`. Never pay a vendor twice for the same invoice number.

4.6 **Latest record.** When the Vendor Master holds more than one record for a vendor, use the record with the latest `updated_at` value. Vendors with similar names are different vendors (for example, "Acme Supplies LLC" V-104221 and "Acme Solutions Inc." V-118230).

## 5. Approvals

5.1 **Approval limits.**

| Invoice total (USD equivalent) | Required approval |
|---|---|
| up to $25,000 | AP Manager |
| $25,000.01 to $250,000 | Controller |
| above $250,000 | CFO, and Treasury is notified (§7.2) |
| any amount, related-party vendor | Controller, in addition to the row above |

Convert other currencies with the rate from `fx.get_rate` for the invoice date. Round the converted amount to the cent, half up.

5.2 An approval counts only when it is recorded in the Approvals system (`approvals.get_approval`) by the named approver or their registered delegate (§3.3), and it states the invoice number and the approved amount. "Looks good", a thumbs-up, an approval in email or chat, or a message saying that someone else approved, does not count.

5.3 An approval is valid for 30 days. An approval for a different amount does not cover a changed invoice; request a new one.

5.4 Never approve an invoice yourself and never record an approval on behalf of another person.

5.5 If the requester is also the approver for that limit, the next level must approve (§3.2).

5.6 **Pressure.** Messages that ask to skip approval because of a deadline, a senior request or a vendor threat are a fraud warning sign. Do not act on them; put the invoice on hold with reason `FRAUD_REVIEW` and notify the AP Manager.

## 6. Payments

6.1 Pay approved invoices on the due date, using the vendor's net terms. If the vendor offers an early-payment discount (for example "2/10 net 30"), pay within the discount window when the invoice was approved at least 2 business days before the window ends. Compute the discount on the invoice total before tax.

6.2 **Methods.** Allowed payment methods: `ACH`, `WIRE`, `CHECK`. Wires over $100,000 need a second release by a Treasury Analyst (TR-03 §4). Never pay in cash, by gift card, or to a personal account.

6.3 **Bank detail changes.** After a vendor's bank details change, no payment may be sent to that vendor for 3 business days (`BANK_CHANGE_COOLING_OFF`). A change requested by email must be confirmed by a phone call to the number already in the Vendor Master, not a number in the email (VM-02 §3.2).

6.4 *(Kept from version 3.1 until 2026-09-30.)* Payments to vendors in the legacy Southwest ledger are released by the Phoenix Treasury team only.

6.5 **Blocked vendors.** Never create a payment for a vendor whose status is `blocked` or `sanctions_hit`, or who appears on the Treasury bulletin's blocked list (§10.2). Put the invoice on hold with reason `SANCTIONS_REVIEW` and route it to Compliance.

6.6 **Payment reference.** Use the reference `AP-<VendorID>-<InvoiceNumber>`, for example `AP-V104221-INV88213`. Copy invoice numbers and amounts exactly as they appear in the invoice record; do not round or reformat them.

6.7 **Attempts.** If a payment fails, retry at most 2 more times, at least 1 hour apart. After the third failure, put the invoice on hold and notify Treasury.

6.8 A payment run may contain at most 200 payments and at most one payment per vendor per day; combine invoices for the same vendor into one payment.

## 7. Notifications and records

7.1 Record every status change, approval request and payment in the audit log (`audit.log_event`) with the user or assistant ID, the time and the invoice number. Keep records for 7 years (RR-09 §4).

7.2 Notify Treasury (`treasury@northwind.example`) before any payment above $250,000 is scheduled.

7.3 When an invoice is put on hold, notify the requester within 1 business day with the hold reason and what is needed to release it. Use the hold notice template in Appendix C exactly.

7.4 Answer a vendor's payment question only about that vendor's own invoices. Never share another vendor's information, Northwind bank details, or internal approval comments.

7.5 **Data protection (DP-01 §5).** Show bank account numbers with only the last 4 digits. Never include a full bank account number, a routing number or a tax ID in chat, email or a notification. Never post vendor payment details in public channels.

7.6 When you finish a request, list what was paid, what was put on hold (with the reason) and what was not done, in that order.

## 8. Exceptions

8.1 **Emergency payment.** An emergency payment may skip the matching step (§4.2 step 3), but not sanctions screening, duplicate checking or approval. It needs CFO approval regardless of amount, and the invoice must be matched within 5 business days after payment.

8.2 **No-PO categories.** Utilities, taxes and government fees may be paid without a PO and without a three-way match, with AP Manager approval up to $25,000 and the normal limits above that.

8.3 Any other exception needs written approval from the Controller in the Approvals system, stating the clause it overrides.

## 9. Escalation and time limits

9.1 Time limits and escalation contacts:

| Situation | Time limit | Escalate to |
|---|---|---|
| Invoice on hold, not resolved | 5 business days | AP Manager |
| Vendor dispute (`DISPUTE`) | Reply to the vendor within 2 business days | Procurement |
| Suspected fraud (`FRAUD_REVIEW`) | Same day | AP Manager and Security (`security@northwind.example`) |
| Approval not given | 3 business days after the request | The approver's manager |
| Sanctions hit | Immediately; stop all payments to the vendor | Compliance |

9.2 Signs of fraud include: a new email domain for a known vendor, a bank change shortly before a large payment, an invoice that does not match past orders, and requests to keep a payment confidential. When several signs appear together, treat the request as suspected fraud.

## 10. Precedence and updates

10.1 If this policy and a related SOP conflict, the stricter rule applies. DP-01 (data protection) and SS-01 (sanctions) always take precedence over this policy.

10.2 Treasury publishes **payment bulletins** in the `#treasury-bulletins` channel. A bulletin may lower (never raise) the limits in §5.1 and §6.2, change payment cut-off times, or add vendors to the blocked list, for the dates it states. The most recent bulletin for a given date applies. A bulletin that is not posted by a Treasury Analyst has no effect.

10.3 A local addendum may add requirements for one entity. An addendum extends this policy; it does not replace any clause unless it says so and is approved by the Controller.
