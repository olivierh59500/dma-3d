package dma3d

import (
	"encoding/binary"
	"testing"

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
		got, found := charToFontIndex(tt.char)
		if !found || got != tt.want {
			t.Errorf("charToFontIndex(%q) = %d, %t; want %d, true", tt.char, got, found, tt.want)
		}
	}
	if _, found := charToFontIndex(' '); found {
		t.Error("charToFontIndex(' ') unexpectedly found a glyph")
	}
}

func TestNewQuadBatch(t *testing.T) {
	vertices, indices := newQuadBatch(2)
	if len(vertices) != 2*verticesPerQuad {
		t.Fatalf("vertices length = %d; want %d", len(vertices), 2*verticesPerQuad)
	}
	wantIndices := []uint16{0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6}
	if len(indices) != len(wantIndices) {
		t.Fatalf("indices length = %d; want %d", len(indices), len(wantIndices))
	}
	for i, want := range wantIndices {
		if indices[i] != want {
			t.Errorf("indices[%d] = %d; want %d", i, indices[i], want)
		}
	}
	for i, vertex := range vertices {
		if vertex.ColorR != 1 || vertex.ColorG != 1 || vertex.ColorB != 1 || vertex.ColorA != 1 {
			t.Errorf("vertex %d color = (%v, %v, %v, %v); want opaque white", i, vertex.ColorR, vertex.ColorG, vertex.ColorB, vertex.ColorA)
		}
	}
}
