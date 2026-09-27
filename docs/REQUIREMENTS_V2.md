# Platform Redesign Requirements (v2)

Status: **In build (Phase 1: schema + wiring).** Produced via a `grill-me` interview on 2026-09-27, branch `v2-app-redesign`.

## Goal

Convert the app from a solo, local video-cutting tool into a shared platform: **upload → share → label/comment → cut (on demand)**, instead of today's upload → auto-detect → auto-cut → label-locally flow.

## 1. Users & Auth

- Closed group. Access stays gated by GCP Identity-Aware Proxy (IAP) + an email allowlist — no signup page, no passwords, no self-service accounts.
- Add a lightweight **user profile** (display name, avatar) layered on top of the IAP-provided email identity, so labels/comments/shares can show a human name instead of a raw email.

## 2. Sharing & Visibility

**Revised 2026-09-27, before build:** kept **fully open**, not private-by-default + explicit-share as originally planned. While the requirements interview was in progress, a commit already shipped (`resolve_within_any_video_root()` / `all_video_owners()` in `app.py`) making every allowlisted user's uploaded videos visible to every other allowlisted user. Confirmed to keep that model rather than build a shares/ACL system on top of it.

- Every video, once uploaded, is visible to every other allowlisted (IAP) user — no per-video visibility toggle, no share/revoke flow, no ACL table.
- Any allowlisted user can: watch, add labels, add comments, export/cut clips.
- Author/uploader/admin distinctions matter only for **editing/deleting** labels and comments (see §4/§5), never for visibility.
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
- Export/cut is **cross-video**: a user can filter by label category + player + comment keyword across **every video on the platform** (visibility is fully open — see §2), and export pulls matching cuts from wherever they live, zipped together.
- A triggered cut becomes part of that source video's **shared clip library** — visible to everyone, same as the source video, not private to whoever triggered the cut.

## 7. Data & Infra

- Move off JSON-file storage to **Postgres (Supabase)** for all relational/structured data: users, calibration profiles, videos, markers, labels, comments, roster, label-category list. (No shares table — see §2.)
- **Video files remain on GCS** exactly as today (gcsfuse mount, existing lifecycle rules) — only structured metadata moves to Postgres.
- **Clean slate migration**: existing JSON-based labels/videos are not imported into the new schema. Old data stays on disk untouched but the new platform starts fresh.
- **Frontend: full React SPA**, replacing the current single-template vanilla-JS UI.
- **Real-time strategy: polling**, not WebSockets/SSE — the SPA periodically re-fetches labels/comments while a video is open (no new real-time infra for v1).
- **Deployment topology:** Flask serves both the compiled React SPA (static files) and the JSON API from **one Cloud Run service**, preserving a single IAP-protected origin. No separate frontend hosting.
- **Deployed as separate Cloud Run services** (`label-ui-v2` / `worker-v2`, via `cloudbuild-v2.yaml`) so v2 development never touches the production `label-ui`/`worker` services real users hit today. Same images, same GCS bucket (video storage is unchanged), same GCP project/region — only the Cloud Run service names, job-queue directory, and deploy trigger differ. One-time manual setup needed (not automatable from a build step):
  1. Create the Secret Manager secret `shot-clipper-db-url` holding the Supabase pooler connection string (`postgresql+psycopg://<user>:<password>@<host>:6543/<database>`), labeled `app=basketball-shot-clipper`.
  2. Create a Cloud Build trigger scoped to the `v2-app-redesign` branch, pointed at `cloudbuild-v2.yaml` (leave the existing `master`-branch trigger pointed at `cloudbuild.yaml` untouched).
  3. After the first deploy, put IAP in front of `label-ui-v2` (same allowlist as the production service) — it starts `--no-allow-unauthenticated` but isn't IAP-protected until that's configured, same as the original service's one-time setup.
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
