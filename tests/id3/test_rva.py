import dataclasses
import logging
import pytest
from pytest import approx
from eyed3.id3 import ID3_V2_3, ID3_V2_4, ID3_V2_2
from eyed3.id3.frames import (
    FrameException, FrameHeader, ReplayGainAdjustmentFrame,
    RelVolAdjFrameV23, RelVolAdjFrameV24, TextFrame, UserTextFrame,
)
from eyed3.id3.tag import Tag, FileInfo
import eyed3


AUDIO = b"\xff\xfb" + b"\x00" * 256


def _synchsafe(n):
    return bytes([(n >> 21) & 0x7f, (n >> 14) & 0x7f,
                  (n >> 7) & 0x7f, n & 0x7f])


def _write_tag(path, frame_bytes):
    with open(path, "wb") as fh:
        fh.write(b"ID3\x04\x00\x00" + _synchsafe(len(frame_bytes))
                 + frame_bytes + AUDIO)


def _text_frame_raw(fid, text):
    payload = b"\x00" + text.encode("latin1")
    return fid + len(payload).to_bytes(4, "big") + b"\x00\x00" + payload


def _new_tag(path, frame_objs, version=ID3_V2_4):
    with open(path, "wb") as fh:
        fh.write(AUDIO)
    tag = Tag()
    tag.file_info = FileInfo(str(path))
    tag.header.version = version
    for frame in frame_objs:
        tag.frame_set[frame.id] = frame
    return tag


def _rgad(track_gain=0, track_peak=0, album_gain=0, album_peak=0):
    f = ReplayGainAdjustmentFrame(
        track_gain=track_gain, track_peak=track_peak,
        album_gain=album_gain, album_peak=album_peak)
    f.header = FrameHeader(b"RGAD", ID3_V2_4)
    return f


def _rva2(channel_type, adjustment, peak=0, identifier=""):
    f = RelVolAdjFrameV24(identifier=identifier, channel_type=channel_type,
                          adjustment=adjustment, peak=peak)
    f.header = FrameHeader(b"RVA2", ID3_V2_4)
    return f


def test_default_v23():
    f = RelVolAdjFrameV23()
    assert f.id == b"RVAD"

    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments()
    f.render()

    f2 = RelVolAdjFrameV23()
    f2.parse(f.data, f.header)

    assert f.adjustments == f2.adjustments
    assert set(dataclasses.astuple(f.adjustments)) == {0}


def test_RelVolAdjFrameV23_invalid_version():
    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments()
    f.render()

    f.header = FrameHeader(f.id, ID3_V2_4)
    with pytest.raises(FrameException):
        f.parse(f.data, f.header)


def test_RelVolAdjFrameV23_outofbounds():
    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments()
    with pytest.raises(ValueError):
        f.adjustments.front_right = 2**16 + 1
        f.render()
    with pytest.raises(ValueError):
        f.adjustments.front_right = -(2**16) - 1
        f.render()

    f.adjustments.front_right = 2**16
    assert f.render()

    f2 = RelVolAdjFrameV23()
    data = bytearray(f.data)
    data[1] = 32
    f.data = bytes(data)
    with pytest.raises(FrameException):
        f2.parse(f.data, f.header)


def test_v23_supported():
    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(
        front_right=-10, front_left=2, front_right_peak=15, front_left_peak=15,
        back_right=54, back_left=-24, back_right_peak=100, back_left_peak=101,
        front_center=10, front_center_peak=15,
        bass=-666, bass_peak=5000,
    )
    assert f.adjustments.has_front_channel
    assert f.adjustments.has_back_channel
    assert f.adjustments.has_front_channel
    assert f.adjustments.has_bass_channel
    assert not f.adjustments.has_master_channel
    assert not f.adjustments.has_other_channel
    assert not f.adjustments.has_back_center_channel

    f.render()

    f2 = RelVolAdjFrameV23()
    f2.parse(f.data, f.header)

    assert f.adjustments == f2.adjustments


def test_v23_unsupported():
    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(
        master=999, master_peak=999, other=333, other_peak=333, back_center=-5, back_center_peak=1,
        front_right=10, front_left=-2, front_right_peak=15, front_left_peak=15,
        back_right=-54, back_left=-24, back_right_peak=100, back_left_peak=101,
        front_center=10, front_center_peak=15,
        bass=666, bass_peak=5000,
    )
    assert f.adjustments.has_front_channel
    assert f.adjustments.has_back_channel
    assert f.adjustments.has_front_channel
    assert f.adjustments.has_bass_channel
    assert f.adjustments.has_master_channel
    assert f.adjustments.has_other_channel
    assert f.adjustments.has_back_center_channel

    f.render()

    f2 = RelVolAdjFrameV23()
    f2.parse(f.data, f.header)

    assert f2.adjustments.has_front_channel
    assert f2.adjustments.has_back_channel
    assert f2.adjustments.has_front_channel
    assert f2.adjustments.has_bass_channel
    assert not f2.adjustments.has_master_channel
    assert not f2.adjustments.has_other_channel
    assert not f2.adjustments.has_back_center_channel

    f.adjustments.master = f.adjustments.master_peak = 0
    f.adjustments.other = f.adjustments.other_peak = 0
    f.adjustments.back_center = f.adjustments.back_center_peak = 0
    assert f.adjustments == f2.adjustments


def test_v23_bounds():
    f = RelVolAdjFrameV23()
    adjustments = dataclasses.asdict(RelVolAdjFrameV23.VolumeAdjustments())

    for a in adjustments.keys():
        values = dict(adjustments)
        for value, raises in [
            (65537, True), (-65537, True),
            (65536, False), (-65536, False),
            (32769, False), (-32768, False),
            (777, False), (-999, False),
            (0, False), (-0, False),
        ]:
            values[a] = value
            if raises:
                with pytest.raises(ValueError):
                    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(**values)
                    f.render()
            else:
                f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(**values)
                f.render()

        assert dataclasses.asdict(f.adjustments)[a] == value


def test_v23_optionals():
    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(
        front_right=-10, front_left=2, front_right_peak=0, front_left_peak=0,
    )
    f.render()
    assert len(f.data) == 10

    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(
        front_right=-10, front_left=2, front_right_peak=0, front_left_peak=0,
        bass=-666, bass_peak=5000,
    )
    f.render()
    assert len(f.data) == 26

    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(
        front_right=-10, front_left=2, front_right_peak=0, front_left_peak=0,
        back_right=54, back_left=-24, back_right_peak=100, back_left_peak=101,
        front_center=10, front_center_peak=15,
    )
    f.render()
    assert len(f.data) == 22

    f = RelVolAdjFrameV23()
    f.adjustments = RelVolAdjFrameV23.VolumeAdjustments(
        front_right=-10, front_left=2, front_right_peak=0, front_left_peak=0,
        back_right=54, back_left=-24, back_right_peak=100, back_left_peak=101,
    )
    f.render()
    assert len(f.data) == 18


def test_default_v24():
    f = RelVolAdjFrameV24()
    assert f.id == b"RVA2"

    f.channel_type = RelVolAdjFrameV24.CHANNEL_TYPE_MASTER
    f.adjustment = -6.3
    f.peak = 666
    f.render()

    f2 = RelVolAdjFrameV24()
    f2.parse(f.data, f.header)
    assert f.adjustment == approx(-6.3)
    assert f2.peak == 666


def test_RVAD_RVA2(audiofile):
    # RVAD -> *RVA2
    audiofile.initTag(version=ID3_V2_3)
    audiofile.tag.frame_set[b"RVAD"] = RelVolAdjFrameV23()
    assert audiofile.tag.frame_set[b"RVAD"][0].adjustments is None
    adj = RelVolAdjFrameV23.VolumeAdjustments(front_left=20, front_right=19,
                                              back_left=-1, bass_peak=1024)
    audiofile.tag.frame_set[b"RVAD"][0].adjustments = adj

    # Convert to RVA2
    audiofile.tag.version = ID3_V2_4
    rva2_frames = {frame.channel_type: frame for frame in audiofile.tag.frame_set[b"RVA2"]}
    assert len(rva2_frames) == 4
    assert set(rva2_frames.keys()) == {RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT,
                                       RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_RIGHT,
                                       RelVolAdjFrameV24.CHANNEL_TYPE_BACK_LEFT,
                                       RelVolAdjFrameV24.CHANNEL_TYPE_BASS}
    assert rva2_frames[RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_RIGHT].adjustment == approx(0.037109375)
    assert rva2_frames[RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT].adjustment == approx(0.0390625)
    assert rva2_frames[RelVolAdjFrameV24.CHANNEL_TYPE_BACK_LEFT].adjustment == approx(-0.001953125)
    assert rva2_frames[RelVolAdjFrameV24.CHANNEL_TYPE_BASS].adjustment == 0
    assert rva2_frames[RelVolAdjFrameV24.CHANNEL_TYPE_BASS].peak == 1024

    # RVA2 --> RVAD
    audiofile.initTag(version=ID3_V2_4)
    assert len(audiofile.tag.frame_set) == 0
    for frame in rva2_frames.values():
        if b"RVA2" not in audiofile.tag.frame_set:
            audiofile.tag.frame_set[b"RVA2"] = frame
        else:
            audiofile.tag.frame_set[b"RVA2"].append(frame)
    assert len(audiofile.tag.frame_set) == 1
    assert len(audiofile.tag.frame_set[b"RVA2"]) == 4

    audiofile.tag.version = ID3_V2_3
    assert len(audiofile.tag.frame_set) == 1
    assert len(audiofile.tag.frame_set[b"RVAD"]) == 1
    assert audiofile.tag.frame_set[b"RVAD"][0].adjustments == \
        RelVolAdjFrameV23.VolumeAdjustments(front_left=20, front_right=19,back_left=-1,
                                            bass_peak=1024)


def test_RelVolAdjFrameV24_channel_type():
    for valid in range(9):
        RelVolAdjFrameV24().channel_type = valid
    for invalid in (-1, 9):
        with pytest.raises(ValueError):
            RelVolAdjFrameV24().channel_type = invalid


def test_RelVolAdjFrameV24_render_invalid_peak():
    rva2 = RelVolAdjFrameV24()
    rva2.peak = 2**32 - 1
    assert rva2.render()

    rva2.peak = 2**32
    with pytest.raises(ValueError):
        rva2.render()
    rva2.peak = 2**64
    with pytest.raises(ValueError):
        rva2.render()


# ---------------------------------------------------------------------------
# RGAD (non-standard Replay Gain Adjustment) frame support
# ---------------------------------------------------------------------------
def test_ReplayGainAdjustmentFrame_roundtrip():
    f = ReplayGainAdjustmentFrame(
        track_gain=-3225, track_peak=12345,
        album_gain=-1536, album_peak=10000)
    f.header = FrameHeader(b"RGAD", ID3_V2_4)
    raw = f.render()

    f2 = ReplayGainAdjustmentFrame(b"RGAD")
    f2.parse(raw[FrameHeader(b"RGAD", ID3_V2_4).size:], f.header)
    assert (f2.track_gain, f2.track_peak, f2.album_gain, f2.album_peak) == \
           (-3225, 12345, -1536, 10000)


def test_ReplayGainAdjustmentFrame_unsigned_peaks():
    # Peaks are unsigned 16-bit; values above 32767 must survive a round-trip.
    f = ReplayGainAdjustmentFrame(track_gain=-1, track_peak=50000,
                                  album_gain=-2, album_peak=65535)
    f.header = FrameHeader(b"RGAD", ID3_V2_4)
    raw = f.render()

    f2 = ReplayGainAdjustmentFrame(b"RGAD")
    f2.parse(raw[FrameHeader(b"RGAD", ID3_V2_4).size:], f.header)
    assert f2.track_peak == 50000
    assert f2.album_peak == 65535
    assert f2.track_gain == -1 and f2.album_gain == -2

    with pytest.raises(ValueError):
        bad = ReplayGainAdjustmentFrame(track_peak=65536)
        bad.header = FrameHeader(b"RGAD", ID3_V2_4)
        bad.render()


def test_ReplayGainAdjustmentFrame_sparse():
    # Only track gain present: one flag bit, one value.
    f = ReplayGainAdjustmentFrame(track_gain=-100)
    f.header = FrameHeader(b"RGAD", ID3_V2_4)
    raw = f.render()

    f2 = ReplayGainAdjustmentFrame(b"RGAD")
    f2.parse(raw[FrameHeader(b"RGAD", ID3_V2_4).size:], f.header)
    assert f2.track_gain == -100
    assert (f2.track_peak, f2.album_gain, f2.album_peak) == (0, 0, 0)


def test_ReplayGainAdjustmentFrame_malformed():
    f = ReplayGainAdjustmentFrame(b"RGAD")
    f.header = FrameHeader(b"RGAD", ID3_V2_4)
    # Flags promise four values but only one byte follows.
    with pytest.raises(FrameException):
        f.parse(b"\x00\x0f\x00", f.header)
    with pytest.raises(FrameException):
        f.parse(b"", f.header)


def test_ReplayGainAdjustmentFrame_wrong_id_and_version():
    with pytest.raises(ValueError):
        ReplayGainAdjustmentFrame(b"XXXX")
    f = ReplayGainAdjustmentFrame(track_gain=1)
    f.header = FrameHeader(b"RGAD", ID3_V2_3)
    with pytest.raises(ValueError):
        f.render()


def test_ReplayGainAdjustmentFrame_registered_nonstandard():
    from eyed3.id3.frames import NONSTANDARD_ID3_FRAMES
    assert NONSTANDARD_ID3_FRAMES[b"RGAD"][2] is ReplayGainAdjustmentFrame


# ---------------------------------------------------------------------------
# v2.4 -> v2.3 save behavior with gain frames
# ---------------------------------------------------------------------------
def test_rgad_save_v23_mirrors_track_to_front_channels(tmp_path):
    path = tmp_path / "rgad.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Artist One"),
        TextFrame(b"TIT2", "Title One"),
        _rgad(track_gain=-3225, track_peak=12345,
              album_gain=-1536, album_peak=10000),
    ])
    tag.save(filename=str(path), version=ID3_V2_3)

    reloaded = eyed3.load(str(path))
    assert reloaded.tag.version == ID3_V2_3
    assert reloaded.tag.artist == "Artist One"
    assert reloaded.tag.title == "Title One"
    assert b"RGAD" not in reloaded.tag.frame_set
    adj = reloaded.tag.frame_set[b"RVAD"][0].adjustments
    assert adj.front_left == adj.front_right == -3225
    assert adj.front_left_peak == adj.front_right_peak == 12345
    # Album values have no RVAD counterpart; everything else stays zero.
    assert all(getattr(adj, n) == 0 for n in dataclasses.asdict(adj)
               if n not in ("front_left", "front_right",
                            "front_left_peak", "front_right_peak"))


def test_rgad_album_drop_warns(tmp_path, caplog):
    path = tmp_path / "rgad_album.mp3"
    tag = _new_tag(path, [_rgad(album_gain=-1536, album_peak=10000)])
    with caplog.at_level(logging.WARNING, logger="eyed3"):
        tag.save(filename=str(path), version=ID3_V2_3)
    assert any("album gain/peak" in rec.message and "dropped" in rec.message
               for rec in caplog.records)


def test_raw_rgad_on_disk_parses_and_saves(tmp_path):
    # End-to-end: a raw RGAD frame written by another tool must parse into a
    # typed frame instead of an UnknownFrame, then save to v2.3.
    path = tmp_path / "raw_rgad.mp3"
    rgad = (b"\x00\x0f" + (-3225).to_bytes(2, "big", signed=True)
            + (12345).to_bytes(2, "big")
            + (-1536).to_bytes(2, "big", signed=True)
            + (10000).to_bytes(2, "big"))
    frames_bytes = _text_frame_raw(b"TPE1", "Raw Artist")
    frames_bytes += b"RGAD" + len(rgad).to_bytes(4, "big") + b"\x00\x00" + rgad
    _write_tag(str(path), frames_bytes)

    audio = eyed3.load(str(path))
    assert isinstance(audio.tag.frame_set[b"RGAD"][0],
                      ReplayGainAdjustmentFrame)
    parsed = audio.tag.frame_set[b"RGAD"][0]
    assert (parsed.track_gain, parsed.track_peak,
            parsed.album_gain, parsed.album_peak) == \
           (-3225, 12345, -1536, 10000)

    audio.tag.save(version=ID3_V2_3)
    reloaded = eyed3.load(str(path))
    assert reloaded.tag.artist == "Raw Artist"
    assert b"RGAD" not in reloaded.tag.frame_set
    assert b"RVAD" in reloaded.tag.frame_set


def test_malformed_rgad_does_not_break_tag(tmp_path, caplog):
    path = tmp_path / "malformed.mp3"
    bad_rgad = b"\x00\x0f\x00"
    frames_bytes = _text_frame_raw(b"TPE1", "Good Artist")
    frames_bytes += (b"RGAD" + len(bad_rgad).to_bytes(4, "big")
                     + b"\x00\x00" + bad_rgad)
    _write_tag(str(path), frames_bytes)

    audio = eyed3.load(str(path))
    assert audio.tag.artist == "Good Artist"
    assert b"RGAD" not in audio.tag.frame_set
    # The neighboring good frame must still save cleanly.
    audio.tag.save(version=ID3_V2_3)
    assert eyed3.load(str(path)).tag.artist == "Good Artist"


def test_rva2_peak_out_of_range_clamps(tmp_path, caplog):
    path = tmp_path / "bigpeak.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Peak Artist"),
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT, -2.0, 2 ** 20),
    ])
    with caplog.at_level(logging.WARNING, logger="eyed3"):
        tag.save(filename=str(path), version=ID3_V2_3)
    assert any("clamping to 65535" in rec.message for rec in caplog.records)

    adj = eyed3.load(str(path)).tag.frame_set[b"RVAD"][0].adjustments
    assert adj.front_left_peak == 65535
    assert adj.front_left == -1024


def test_rva2_fractional_adjustment_rounds(tmp_path, caplog):
    path = tmp_path / "float.mp3"
    tag = _new_tag(path, [
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT,
              0.037109375 * 1.0001, 0),
    ])
    with caplog.at_level(logging.WARNING, logger="eyed3"):
        tag.save(filename=str(path), version=ID3_V2_3)
    assert any("Rounding ReplayGain" in rec.message for rec in caplog.records)
    adj = eyed3.load(str(path)).tag.frame_set[b"RVAD"][0].adjustments
    assert adj.front_left == 19


def test_rva2_master_mirrors_to_front_channels(tmp_path):
    path = tmp_path / "master.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Master Artist"),
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_MASTER, -3226 / 512, 2 ** 20),
    ])
    tag.save(filename=str(path), version=ID3_V2_3)

    adj = eyed3.load(str(path)).tag.frame_set[b"RVAD"][0].adjustments
    assert adj.front_left == adj.front_right == -3226
    assert adj.front_left_peak == adj.front_right_peak == 65535


def test_rva2_master_dropped_when_fronts_occupied(tmp_path, caplog):
    path = tmp_path / "master_conflict.mp3"
    tag = _new_tag(path, [
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT, 1.0, 10),
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_RIGHT, 0.5, 5),
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_MASTER, -6.3, 999),
    ])
    with caplog.at_level(logging.WARNING, logger="eyed3"):
        tag.save(filename=str(path), version=ID3_V2_3)
    assert any("master" in rec.message and "dropping" in rec.message.lower()
               for rec in caplog.records)

    adj = eyed3.load(str(path)).tag.frame_set[b"RVAD"][0].adjustments
    # Explicit channels are untouched by the master conflict.
    assert adj.front_left == 512 and adj.front_left_peak == 10
    assert adj.front_right == 256 and adj.front_right_peak == 5


def test_rgad_and_rva2_merge_precedence(tmp_path):
    path = tmp_path / "mixed.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Mixed Artist"),
        _rgad(track_gain=-3225, track_peak=111),
        _rva2(RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT, 0.0390625, 15),
    ])
    tag.save(filename=str(path), version=ID3_V2_3)

    adj = eyed3.load(str(path)).tag.frame_set[b"RVAD"][0].adjustments
    # Explicit RVA2 front-left wins; the RGAD mirror survives on the right.
    assert adj.front_left == 20 and adj.front_left_peak == 15
    assert adj.front_right == -3225 and adj.front_right_peak == 111


def test_txxx_replaygain_frames_preserved_untouched(tmp_path):
    path = tmp_path / "txxx.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "TXXX Artist"),
        UserTextFrame(b"TXXX", description="replaygain_track_gain",
                      text="-6.30 dB"),
        UserTextFrame(b"TXXX", description="replaygain_album_gain",
                      text="-3.10 dB"),
    ])
    tag.save(filename=str(path), version=ID3_V2_3)

    reloaded = eyed3.load(str(path)).tag
    txxx = {f.description: f.text for f in reloaded.user_text_frames}
    assert txxx == {
        "replaygain_track_gain": "-6.30 dB",
        "replaygain_album_gain": "-3.10 dB",
    }
    assert b"RVAD" not in reloaded.frame_set


def test_repeated_v23_saves_are_stable(tmp_path):
    path = tmp_path / "repeat.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Repeat"),
        _rgad(track_gain=-3225, track_peak=12345),
    ])
    for _ in range(3):
        tag.save(filename=str(path), version=ID3_V2_3)
        tag = eyed3.load(str(path)).tag
    adj = tag.frame_set[b"RVAD"][0].adjustments
    assert adj.front_left == adj.front_right == -3225
    assert adj.front_left_peak == adj.front_right_peak == 12345


def test_rgad_v23_then_v24_roundtrip(tmp_path):
    path = tmp_path / "roundtrip.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Round Trip"),
        _rgad(track_gain=-512, track_peak=256),
    ])
    tag.save(filename=str(path), version=ID3_V2_3)
    assert b"RVAD" in eyed3.load(str(path)).tag.frame_set

    reloaded = eyed3.load(str(path))
    reloaded.tag.save(version=ID3_V2_4)
    again = eyed3.load(str(path))
    assert again.tag.version == ID3_V2_4
    assert b"RVA2" in again.tag.frame_set
    rva2 = {f.channel_type: f for f in again.tag.frame_set[b"RVA2"]}
    assert rva2[RelVolAdjFrameV24.CHANNEL_TYPE_FRONT_LEFT].adjustment == \
        approx(-1.0)


def test_plain_tag_save_has_no_gain_frames(tmp_path):
    path = tmp_path / "plain.mp3"
    tag = _new_tag(path, [
        TextFrame(b"TPE1", "Plain Artist"),
        TextFrame(b"TIT2", "Plain Title"),
    ])
    tag.save(filename=str(path), version=ID3_V2_3)
    reloaded = eyed3.load(str(path)).tag
    assert reloaded.artist == "Plain Artist"
    assert reloaded.title == "Plain Title"
    assert b"RVAD" not in reloaded.frame_set
    assert b"RGAD" not in reloaded.frame_set
