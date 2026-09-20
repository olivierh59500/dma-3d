// Package mobile exposes the demo to ebitenmobile.
package mobile

import (
	"fmt"

	enginemobile "github.com/hajimehoshi/ebiten/v2/mobile"

	dma3d "dma-3d/dck"
)

func init() {
	game, err := dma3d.NewGame()
	if err != nil {
		panic(fmt.Sprintf("initialize DMA 3D demo: %v", err))
	}
	enginemobile.SetGame(game)
}

// Dummy forces gomobile to include this package in the Android binding.
func Dummy() {}
