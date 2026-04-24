from loguru import logger


def terminated_check(episode_status: dict, game_over: bool):
    if game_over and (
        episode_status['lap_complete']
        or
        episode_status['crashed']
        or
        episode_status['reversed']
        or
        episode_status['off_track']
    ):
        return True
    return False


def truncated_check(episode_status: dict, game_over: bool):
    terminated = terminated_check(episode_status, game_over)
    # time_out or immobilized
    truncated = (game_over and not terminated)
    if truncated:
        status = [k for k, v in episode_status.items() if v]
        if not episode_status['immobilized'] and not episode_status['time_up']:
            logger.warning(
                f'Expected immibilized or time_up status for truncated episode.'
                f'Instead got {status}.'
                f'Restart deepracer to prevent unexpected behavior.'
            )
    return truncated
