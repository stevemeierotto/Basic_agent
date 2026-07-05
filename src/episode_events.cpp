/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — execution-domain episode events (E2-C2)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */
#include "episode_events.h"

namespace Thoth {

nlohmann::json EpisodeCompleted::toJson() const {
    return {{"plan_id", plan_id},
            {"goal", goal},
            {"terminal_state", terminal_state},
            {"final_success_score", final_success_score},
            {"completed_at_ms", completed_at_ms},
            {"run_id", run_id},
            {"env_hash", env_hash},
            {"plan_snapshot", plan_snapshot},
            {"trajectory_snapshot", trajectory_snapshot}};
}

} // namespace Thoth
