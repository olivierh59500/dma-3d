package main

import (
	"log"

	"github.com/hajimehoshi/ebiten/v2"

	dma3d "dma-3d"
)

func main() {
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("DMA 3D Demo (Go/Ebitengine/ym-player)")

	game, err := dma3d.NewGame()
	if err != nil {
		log.Fatal(err)
	}
	defer game.Cleanup()

	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
