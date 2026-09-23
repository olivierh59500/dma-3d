package dma3d

import (
	"github.com/olivierh59500/democonstructionkit/presets"
	"testing"
)

func legacyAtlasIndex(ch rune) (int, bool) {
	if ch >= '0' && ch <= '9' {
		return 16 + int(ch-'0'), true
	}
	if ch >= 'A' && ch <= 'Z' {
		return 33 + int(ch-'A'), true
	}

	// Font layout (6 rows of 10 characters)
	switch ch {
	case '!':
		return 1, true
	case '"':
		return 2, true
	case '\'':
		return 7, true
	case '(':
		return 8, true
	case ')':
		return 9, true
	case ',':
		return 12, true
	case '-':
		return 13, true
	case '.':
		return 14, true
	case ':':
		return 27, true
	case ';':
		return 28, true
	case '?':
		return 31, true
	default:
		return 0, false
	}
}
func TestSharedAtlasIndicesMatchOriginalAlphabet(t *testing.T) {
	lookup, err := presets.TileLookup("dma-3d", true)
	if err != nil {
		t.Fatal(err)
	}
	for r := rune(0); r < 256; r++ {
		got, ok := lookup(r)
		want, found := legacyAtlasIndex(r)
		if got != want || ok != found {
			t.Fatalf("rune %U: got (%d,%t), want (%d,%t)", r, got, ok, want, found)
		}
	}
}
