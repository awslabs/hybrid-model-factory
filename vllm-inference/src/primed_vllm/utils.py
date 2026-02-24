def parse_override_pattern(pattern: str):
    """
    Converts a string pattern like "*-*-*-M2-M2-BMF" (where * denotes attention, M2 Mamba2, BMF BMOJO-F)
    into a dictionary mapping layer index to layer type, e.g., 
    {
        0: *,
        1: *,
        2: *,
        3: M2,
        4: M2,
        5: BMF
    }

    Symbol definitions:
        * -> Attention
        M2 -> Mamba2
        BMF -> BMOJO-F
        GDN -> GatedDeltaNet

        SWA -> Sliding window attention
    """
    split = pattern.split("-")
    parsed_pattern = {i: s for i, s in enumerate(split)}

    return parsed_pattern

class Symbols:
    MAMBA = 'M2'
    ATTENTION = '*'
    BMOJO_F = 'BMF'
    GDN = 'GDN'
    GKA = 'GKA'
    SWA = 'SWA'
    VALID = {MAMBA, ATTENTION, BMOJO_F, GDN, GKA, SWA}

    # Map legacy/alternative ssm_mixer strings to canonical symbols
    SSM_MIXER_MAP = {
        "mamba2": 'M2',
        "gated_deltanet": 'GDN',
        "gdn": 'GDN',
        "gka": 'GKA',
    }


def pattern_has_symbols(pattern: str, bmojo_config: dict | None = None) -> set[str]:
    """Return the set of canonical Symbols present in a raw hybrid_override_pattern string.

    Optionally inspects bmojo_config.ssm_mixer to resolve BMOJO-F sub-layer types.
    """
    allocation = parse_override_pattern(pattern)
    symbols = set(allocation.values())
    if bmojo_config and Symbols.BMOJO_F in symbols:
        mixer = bmojo_config.get("ssm_mixer", "").lower()
        resolved = Symbols.SSM_MIXER_MAP.get(mixer)
        if resolved:
            symbols.add(resolved)
    return symbols


if __name__ == "__main__":
    pattern = "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-M2-M2-M2-M2-M2-M2-M2-M2"
    parsed_pattern = parse_override_pattern(pattern)
    print(parsed_pattern)

