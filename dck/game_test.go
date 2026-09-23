package dma3d

import (
	"encoding/binary"
	"testing"

	"github.com/olivierh59500/democonstructionkit/presets"
	"github.com/olivierh59500/democonstructionkit/sound"
)

func TestAudioSampleRate(t *testing.T) {
	if sampleRate != 48000 {
		t.Fatalf("sampleRate = %d; want 48000", sampleRate)
	}
}

func TestMusicStreamReadProducesStereoWithoutAllocating(t *testing.T) {
	player, err := sound.Open("music.ym", musicData, sound.Options{SampleRate: sampleRate, Loop: true, PCMFormat: sound.PCM16, Gain: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := player.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})

	buffer := make([]byte, 4096*pcmFrameBytes)
	read := func() {
		n, err := player.Read(buffer)
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		if n != len(buffer) {
			t.Fatalf("Read bytes = %d; want %d", n, len(buffer))
		}
	}

	read()
	for i := 0; i < len(buffer); i += pcmFrameBytes {
		left := binary.LittleEndian.Uint16(buffer[i : i+2])
		right := binary.LittleEndian.Uint16(buffer[i+2 : i+4])
		if left != right {
			t.Fatalf("frame %d is not mono duplicated to stereo: %d != %d", i/pcmFrameBytes, left, right)
		}
	}

	if allocations := testing.AllocsPerRun(20, read); allocations != 0 {
		t.Fatalf("Read allocations = %v; want 0", allocations)
	}
}

func TestCharToFontIndex(t *testing.T) {
	lookup, err := presets.TileLookup("dma-3d", true)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		char rune
		want int
	}{
		{char: '0', want: 16},
		{char: '9', want: 25},
		{char: 'A', want: 33},
		{char: 'Z', want: 58},
		{char: '?', want: 31},
	}
	for _, tt := range tests {
		got, found := lookup(tt.char)
		if !found || got != tt.want {
			t.Errorf("charToFontIndex(%q) = %d, %t; want %d, true", tt.char, got, found, tt.want)
		}
	}
	if _, found := lookup(' '); found {
		t.Error("lookup(' ') unexpectedly found a glyph")
	}
}
