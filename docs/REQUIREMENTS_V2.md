# Platform Redesign Requirements (v2)

Status: **Requirements locked, pre-build.** Produced via a `grill-me` interview on 2026-09-27, branch `v2-platform-redesign`.

## Goal

Convert the app from a solo, local video-cutting tool into a shared platform: **upload → share → label/comment → cut (on demand)**, instead of today's upload → auto-detect → auto-cut → label-locally flow.

## 1. Users & Auth

- Closed group. Access stays gated by GCP Identity-Aware Proxy (IAP) + an email allowlist — no signup page, no passwords, no self-service accounts.
- Add a lightweight **user profile** (display name, avatar) layered on top of the IAP-provided email identity, so labels/comments/shares can show a human name instead of a raw email.

## 2. Sharing & Visibility

- Videos are **private by default**.
- Uploader explicitly shares a video with **specific named users** (picked from the allowlist) — not a single global "public" toggle.
- Permission model for a shared viewer:
  - Can: watch, add labels, add comments, export/cut clips they select.
  - Cannot: delete the source video, edit another user's labels/comments, re-share to people the owner hasn't approved.
- Uploader (or an admin) can revoke a user's access later. Revoking access does **not** delete that user's prior labels/comments — they remain as historical record; the revoked user just loses further access.
- No duplicate-upload detection in v1 (two people uploading the same footage separately is tolerated).

## 3. Video → Marker → Clip Model (core architectural pivot)

- **Calibration becomes a reusable per-camera/court profile** (e.g. "Home Court, Camera 1"), not per-video as today. A user selects/creates a profile once; new uploads tagged with an existing profile skip manual calibration.
- On upload, if the video's calibration profile already exists, **auto-detection runs automatically in the background** (existing YOLO pipeline + job queue/worker).
- **No automatic pre-cutting.** Auto-detection drops **unconfirmed suggested markers** (timestamps) onto the video's timeline — it does not cut files and does not auto-apply a label. A human reviews the full video, jumps marker-to-marker ("next marker" navigation), and confirms/edits/dismisses each one.
- Users can also drop **arbitrary custom timestamp markers** anywhere on the video, not limited to auto-detected candidates.
- Auto-detection stays scoped to **shot-attempt/goal candidates only** (matches current YOLO capability). Assist, block, and any other category are always manual — no ML scope expansion in this redesign.
- **Cutting real clip files is a separate, explicit, on-demand action**, triggered later by a user, not automatic on upload or on label.

## 4. Labels

- **Fixed, admin-managed category list**, starting with: **Goal, Assist, Block**. An admin can add/edit/remove categories later as usage reveals what's needed (no hardcoded enum baked into application logic beyond a lookup table).
- Each label also carries an optional **player tag** (free-text against a **global shared roster** — one platform-wide player list, no team grouping/no per-video roster).
- Labels are **per-user attributed, not a single shared value**. Multiple users can independently label the same marker (possibly disagreeing); the marker shows all contributions (e.g. "3× Goal, 1× Block") rather than one edit overwriting another.
- Permissions: any user with view access can add labels. A user can edit/delete only their own labels. The video's uploader and platform admins can **delete** (not edit) any label on that video, for moderation.

## 5. Comments

- Comments are **pinned to a timestamp/marker** on the timeline (not whole-video-only, not time-ranges).
- **Flat list, no threaded replies** in v1.
- Same permission model as labels: own comments editable/deletable by the author; uploader/admin can delete (not edit) any comment for moderation.
- Comment text is **keyword/substring searchable**, and combinable with label filters at export time (e.g. "Goal AND comment contains 'buzzer'").

## 6. Cutting / Export

- Clip boundaries = **fixed padding around a marker** (default e.g. 5s pre / 2s post, configurable), with **overlapping padded ranges auto-merged** into a single clip (no separate manual gap threshold).
- Export/cut is **cross-video**: a user can filter by label category + player + comment keyword across **every video they have access to** (their own uploads + videos shared with them), and export pulls matching cuts from wherever they live, zipped together.
- A triggered cut becomes part of that source video's **shared clip library** — visible to everyone who already has access to the source video, not private to whoever triggered the cut.

## 7. Data & Infra

- Move off JSON-file storage to **Postgres (Supabase)** for all relational/structured data: users, profiles, shares, calibration profiles, markers, labels, comments, roster, label-category list.
- **Video files remain on GCS** exactly as today (gcsfuse mount, existing lifecycle rules) — only structured metadata moves to Postgres.
- **Clean slate migration**: existing JSON-based labels/videos are not imported into the new schema. Old data stays on disk untouched but the new platform starts fresh.
- **Frontend: full React SPA**, replacing the current single-template vanilla-JS UI.
- **Real-time strategy: polling**, not WebSockets/SSE — the SPA periodically re-fetches labels/comments while a video is open (no new real-time infra for v1).
- **Deployment topology:** Flask serves both the compiled React SPA (static files) and the JSON API from **one Cloud Run service**, preserving a single IAP-protected origin. No separate frontend hosting.
- Notifications (email/in-app for shares/comments) are **deferred**, not part of this redesign.
- **Desktop-first** UI; mobile gets best-effort responsiveness, not dedicated optimization, in v1.

## Explicitly Out of Scope (v1)

- Self-service signup / open registration.
- Threaded comment replies.
- Real-time (WebSocket) collaboration.
- Duplicate-video detection.
- Team/roster grouping (players are a flat global list).
- Expanding YOLO auto-detection beyond shot-attempt/goal candidates.
- Migrating existing JSON-based label/video data.
- Email/in-app notifications.
- Dedicated mobile optimization.

## Known Follow-ups / Risks to Track During Build

- Supabase Postgres is hosted outside GCP — won't carry the `app=basketball-shot-clipper` GCP label/budget convention, and may add cross-region latency vs Cloud Run in `asia-southeast1`. Store its credentials in Secret Manager, not in code.
- Per-user attributed labels change the meaning of existing single-value fields (`goal`/`no_goal`, `stars`) from today's dataset-training pipeline (`shot-clipper-build-dataset`, `shot-clipper-train-filter`) — the ML training pipeline's assumption of one label per clip will need reconciling with "many labels per marker" before it can consume v2 data.
- Calibration-profile reuse (per camera/court) is new; today's calibration is strictly per-video-filename — needs a real profile-selection UX at upload time.
