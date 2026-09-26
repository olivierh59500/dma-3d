// Package dma3d implements the Glenz vector demo.
package dma3d

import (
	"bytes"
	originalassets "dma-3d"
	"fmt"
	"github.com/olivierh59500/democonstructionkit/presets"
	"image"
	"image/color"

	kit "github.com/olivierh59500/democonstructionkit"
	"github.com/olivierh59500/democonstructionkit/effects"
	"github.com/olivierh59500/democonstructionkit/scrolling"
	"github.com/olivierh59500/democonstructionkit/sound"
	"github.com/olivierh59500/democonstructionkit/sprites"

	_ "image/png"
	"log"

	"github.com/hajimehoshi/ebiten/v2"

	audio "github.com/olivierh59500/democonstructionkit/sound/output"

	demolayout "dma-3d/dck/internal/layout"
)

const (
	screenWidth   = demolayout.SceneWidth
	screenHeight  = demolayout.SceneHeight
	sampleRate    = 48000
	pcmFrameBytes = 4 // 16-bit little-endian stereo

	glyphWidth  = 64
	glyphHeight = 50
)

var (
	fontData = originalassets.DCKAssetFontData()

	musicData = originalassets.DCKAssetMusicData()
)

type Game struct {
	scrollRenderer     *scrolling.Scrolling
	starfield          *sprites.BatchedSolidField
	mesh               *effects.MorphingMesh
	fontImg            *ebiten.Image
	colorImage         *ebiten.Image
	audioContext       *audio.Context
	audioPlayer        *audio.Player
	musicStream        *sound.Stream
	scrollText         string
	audioInitAttempted bool
	sceneImage         *ebiten.Image
}

func NewGame() (*Game, error) {
	g := &Game{
		scrollText: "                HELLO, BLAH BLAH BLAH, ABCDEF GHIJKL MNOPQ RSTVU WXYZ. 01234 56789     ON ZAPPE....        ",
		sceneImage: ebiten.NewImage(screenWidth, screenHeight),
	}

	if err := g.loadImages(); err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	atlas, err := presets.FontAtlas("dma-3d", g.fontImg)
	if err != nil {
		return nil, err
	}
	config := presets.DMA3DRowColumn(atlas, g.scrollText)
	g.scrollRenderer, err = scrolling.New(scrolling.Config{RowColumn: &config})
	if err != nil {
		return nil, err
	}
	g.colorImage = ebiten.NewImage(1, 1)
	g.colorImage.Fill(color.RGBA{255, 255, 255, 255}) // White with full alpha
	g.mesh, err = effects.NewMorphingMesh(presets.DMA3DMorphMesh(g.colorImage))
	if err != nil {
		return nil, err
	}
	starOptions := presets.DefaultDMA3DStarOptions(screenWidth, screenHeight)
	starOptions.Source = g.colorImage
	starConfig, err := presets.DMA3DStarfield(starOptions)
	if err != nil {
		return nil, err
	}
	g.starfield, err = sprites.NewBatchedSolidField(starConfig)
	if err != nil {
		return nil, err
	}

	return g, nil
}

func (g *Game) initAudio() error {
	g.audioContext = audio.NewContext(sampleRate)
	var err error
	g.musicStream, err = sound.Open("music.ym", musicData, sound.Options{SampleRate: sampleRate, Loop: true, PCMFormat: sound.PCM16, Gain: 1})
	if err != nil {
		return err
	}
	g.audioPlayer, err = g.audioContext.NewPlayer(g.musicStream)
	if err != nil {
		if closeErr := g.musicStream.Close(); closeErr != nil {
			log.Printf("close music stream after audio initialization failure: %v", closeErr)
		}
		g.musicStream = nil
		return err
	}
	g.audioPlayer.SetVolume(0.7)
	g.audioPlayer.Play()
	return nil
}

func (g *Game) loadImages() error {
	img, _, err := image.Decode(bytes.NewReader(fontData))
	if err != nil {
		return fmt.Errorf("failed to load TCB font: %w", err)
	}
	g.fontImg = ebiten.NewImageFromImage(img)
	if g.fontImg.Bounds().Dx() < 10*glyphWidth || g.fontImg.Bounds().Dy() < 6*glyphHeight {
		return fmt.Errorf("TCB font dimensions are %s; want at least %dx%d", g.fontImg.Bounds(), 10*glyphWidth, 6*glyphHeight)
	}

	return nil
}

func (g *Game) Update() error {
	// On Android, NewGame runs while the native library is loading. Opening the
	// audio device here ensures that the Activity and EbitenView already exist.
	if !g.audioInitAttempted {
		g.audioInitAttempted = true
		if err := g.initAudio(); err != nil {
			log.Printf("audio disabled: %v", err)
		}
	}

	// DCK owns the glyph, row-wave and column-cosine transport.
	if err := g.scrollRenderer.Update(kit.Frame{}); err != nil {
		return err
	}

	if err := g.mesh.Update(kit.Frame{}); err != nil {
		return err
	}
	return g.starfield.Update(kit.Frame{})
}

func (g *Game) Draw(screen *ebiten.Image) {
	scene := g.sceneImage
	scene.Fill(color.Black)

	// Draw starfield first, but skip center area where 3D objects are
	g.starfield.Draw(scene)

	// Draw 3D objects on top
	g.mesh.Draw(scene)

	// Draw scrolling text
	g.scrollRenderer.Draw(scene)

	// Preserve the original 4:3 canvas on wide mobile screens and center it in
	// the available logical surface instead of stretching the demo.
	if screen.Bounds().Dx() > screenWidth {
		screen.Fill(color.Black)
	}
	var op ebiten.DrawImageOptions
	op.GeoM.Translate(float64((screen.Bounds().Dx()-screenWidth)/2), 0)
	screen.DrawImage(scene, &op)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return demolayout.LogicalWidth(outsideWidth, outsideHeight), screenHeight
}

func (g *Game) Cleanup() {
	if g.mesh != nil {
		g.mesh.Close()
	}
	if g.starfield != nil {
		g.starfield.Close()
	}
	if g.scrollRenderer != nil {
		g.scrollRenderer.Close()
	}
	if g.audioPlayer != nil {
		if err := g.audioPlayer.Close(); err != nil {
			log.Printf("close audio player: %v", err)
		}
	}
	if g.musicStream != nil {
		if err := g.musicStream.Close(); err != nil {
			log.Printf("close music stream: %v", err)
		}
	}
}
