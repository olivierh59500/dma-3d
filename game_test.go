package dma3d

import (
	"encoding/binary"
	"testing"
)

func TestAudioSampleRate(t *testing.T) {
	if sampleRate != 48000 {
		t.Fatalf("sampleRate = %d; want 48000", sampleRate)
	}
}

func TestYMPlayerReadProducesStereoWithoutAllocating(t *testing.T) {
	player, err := NewYMPlayer(musicData, sampleRate, true)
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
