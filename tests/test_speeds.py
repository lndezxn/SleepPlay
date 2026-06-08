from sleepplay.speeds import LinearSpeedMapper, SensitiveSpeedMapper, SpeedContext


def test_still_score_maps_to_max_speed() -> None:
    mapper = LinearSpeedMapper(
        still_score=2.0,
        motion_score=22.0,
        min_speed=1.0,
        max_speed=16.0,
        smoothing_window=1,
    )

    assert mapper.map_score(2.0) == 16.0


def test_motion_score_maps_to_min_speed() -> None:
    mapper = LinearSpeedMapper(
        still_score=2.0,
        motion_score=22.0,
        min_speed=1.0,
        max_speed=16.0,
        smoothing_window=1,
    )

    assert mapper.map_score(22.0) == 1.0


def test_middle_score_interpolates_linearly() -> None:
    mapper = LinearSpeedMapper(
        still_score=2.0,
        motion_score=22.0,
        min_speed=1.0,
        max_speed=16.0,
        smoothing_window=1,
    )

    assert mapper.map_score(12.0) == 8.5


def test_scores_clamp_to_speed_bounds() -> None:
    mapper = LinearSpeedMapper(
        still_score=2.0,
        motion_score=22.0,
        min_speed=1.0,
        max_speed=16.0,
        smoothing_window=1,
    )

    assert mapper.map_score(-10.0) == 16.0
    assert mapper.map_score(100.0) == 1.0


def test_sensitive_mapper_slows_down_small_motion_more_than_linear() -> None:
    linear_mapper = LinearSpeedMapper(
        still_score=1.0,
        motion_score=10.0,
        min_speed=1.0,
        max_speed=16.0,
        smoothing_window=1,
    )
    sensitive_mapper = SensitiveSpeedMapper(
        still_score=1.0,
        motion_score=10.0,
        min_speed=1.0,
        max_speed=16.0,
        smoothing_window=1,
        sensitivity=4.0,
    )

    assert sensitive_mapper.map_score(2.0) < linear_mapper.map_score(2.0)


def test_mapper_smooths_score_series_before_mapping() -> None:
    unsmoothed_mapper = LinearSpeedMapper(
        still_score=0.0,
        motion_score=12.0,
        min_speed=1.0,
        max_speed=13.0,
        smoothing_window=1,
    )
    smoothed_mapper = LinearSpeedMapper(
        still_score=0.0,
        motion_score=12.0,
        min_speed=1.0,
        max_speed=13.0,
        smoothing_window=3,
    )
    context = SpeedContext(
        times=[0.0, 1.0, 2.0],
        scores=[0.0, 12.0, 0.0],
        frame_interval_seconds=1.0,
    )

    assert unsmoothed_mapper.map_scores(context) == [13.0, 1.0, 13.0]
    assert smoothed_mapper.map_scores(context) == [7.0, 9.0, 7.0]
