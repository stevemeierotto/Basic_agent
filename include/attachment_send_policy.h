/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — ALP-C revision decision tree (pure policy)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#ifndef THOTH_ATTACHMENT_SEND_POLICY_H
#define THOTH_ATTACHMENT_SEND_POLICY_H

#include <cstdint>
#include <optional>
#include <string>

namespace Thoth {
namespace AttachmentSendPolicy {

enum class SendAction {
    Create,
    NewRevision,
    NoOp,
    LinkOnly,
    Conflict,
    Retry,
};

struct CommittedRevision {
    std::string revision_id;
    std::string content_hash;
    std::int64_t indexed_at_ms = 0;
};

struct InFlightRevision {
    std::string revision_id;
    std::string content_hash;
};

struct PolicyInput {
    bool document_exists = false;
    std::optional<CommittedRevision> committed;
    std::optional<InFlightRevision> in_flight;
    std::string content_hash;
    std::int64_t local_source_mtime_sec = 0;
    bool force_replace = false;
    bool last_revision_failed = false;
};

struct PolicyResult {
    SendAction action = SendAction::Create;
    std::string reason;
};

inline std::string actionToString(SendAction action) {
    switch (action) {
    case SendAction::Create:
        return "create";
    case SendAction::NewRevision:
        return "new_revision";
    case SendAction::NoOp:
        return "no_op";
    case SendAction::LinkOnly:
        return "link_only";
    case SendAction::Conflict:
        return "conflict";
    case SendAction::Retry:
        return "retry";
    }
    return "create";
}

inline PolicyResult evaluate(const PolicyInput& input) {
    PolicyResult out;
    if (!input.document_exists) {
        out.action = SendAction::Create;
        out.reason = "new_document_slot";
        return out;
    }

    if (input.committed && input.committed->content_hash == input.content_hash) {
        out.action = SendAction::NoOp;
        out.reason = "hash_matches_committed";
        return out;
    }

    if (input.in_flight && input.in_flight->content_hash == input.content_hash) {
        out.action = SendAction::NoOp;
        out.reason = "hash_matches_in_flight";
        return out;
    }

    if (input.last_revision_failed) {
        out.action = SendAction::Retry;
        out.reason = "retry_after_failed_revision";
        return out;
    }

    const bool hash_differs =
        !input.committed || input.committed->content_hash != input.content_hash;

    if (!hash_differs) {
        out.action = SendAction::NoOp;
        out.reason = "hash_matches";
        return out;
    }

    if (input.committed && input.local_source_mtime_sec > 0
        && input.committed->indexed_at_ms > 0) {
        const std::int64_t committed_sec = input.committed->indexed_at_ms / 1000;
        if (input.local_source_mtime_sec <= committed_sec) {
            if (input.force_replace) {
                out.action = SendAction::NewRevision;
                out.reason = "force_replace";
                return out;
            }
            out.action = SendAction::Conflict;
            out.reason = "local_mtime_not_newer";
            return out;
        }
    }

    out.action = SendAction::NewRevision;
    out.reason = "content_changed";
    return out;
}

} // namespace AttachmentSendPolicy
} // namespace Thoth

#endif // THOTH_ATTACHMENT_SEND_POLICY_H
