// Command video exports the complete game canvas and its own audio.
package main

import (
	"flag"
	"log"
	"time"

	demo "dma-3d/dck"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/olivierh59500/democonstructionkit/video"
)

func main() {
	config := video.Config{Output: "dma-3d.mp4", Title: "DMA 3D", Width: 640, Height: 480, FPS: 60, TPS: 60, SampleRate: 48000, Duration: 3 * time.Minute}
	config.Flags(flag.CommandLine)
	flag.Parse()
	if err := video.Run(config, func() (ebiten.Game, error) {
		return demo.NewGame()
	}); err != nil {
		log.Fatal(err)
	}
}
