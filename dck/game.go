// Package dma3d implements the Glenz vector demo.
package dma3d

import (
	"bytes"
	"cmp"
	originalassets "dma-3d"
	"fmt"
	"github.com/olivierh59500/democonstructionkit/presets"
	"image"
	"image/color"

	kit "github.com/olivierh59500/democonstructionkit"
	"github.com/olivierh59500/democonstructionkit/scrolling"
	"github.com/olivierh59500/democonstructionkit/sound"
	"github.com/olivierh59500/democonstructionkit/sprites"

	_ "image/png"
	"log"
	"math"
	"slices"

	"github.com/hajimehoshi/ebiten/v2"

	audio "github.com/olivierh59500/democonstructionkit/sound/output"

	demolayout "dma-3d/dck/internal/layout"
)

const (
	screenWidth   = demolayout.SceneWidth
	screenHeight  = demolayout.SceneHeight
	sampleRate    = 48000
	pcmFrameBytes = 4 // 16-bit little-endian stereo

	glyphWidth         = 64
	glyphHeight        = 50
	fontGlyphCount     = 60
	scrollWorkPadding  = 512
	scrollLineHeight   = 2
	scrollLineCount    = glyphHeight / scrollLineHeight
	scrollColumnWidth  = 16
	scrollColumnCount  = screenWidth / scrollColumnWidth
	scrollWaveDataSize = 1191
	fullRotation       = 2 * math.Pi
)

var (
	fontData = originalassets.DCKAssetFontData()

	musicData = originalassets.DCKAssetMusicData()
)

type Vector3 struct{ X, Y, Z float64 }
type Triangle struct {
	V1, V2, V3 int
	Color      color.RGBA
}
type TriangleWithDepth struct {
	Triangle
	Depth float64
}
type Game struct {
	scrollRenderer     *scrolling.Scrolling
	starfield          *sprites.BatchedSolidField
	drawTriOp          *ebiten.DrawTrianglesOptions
	fontImg            *ebiten.Image
	colorImage         *ebiten.Image
	audioContext       *audio.Context
	audioPlayer        *audio.Player
	musicStream        *sound.Stream
	vertices           [][]Vector3
	currentVertices    []Vector3
	triangles          []Triangle
	transformedVerts   []Vector3
	trianglesDepth     []TriangleWithDepth
	rotationX          float64
	rotationY          float64
	rotationZ          float64
	morphTimer         float64
	currentShape       int
	targetShape        int
	scrollText         string
	audioInitAttempted bool
	sceneImage         *ebiten.Image
	projectedVerts     []projectedVertex
	triangleVertices   [3]ebiten.Vertex
	triangleIndices    []uint16
}

type projectedVertex struct {
	X, Y float32
}

func NewGame() (*Game, error) {
	g := &Game{
		drawTriOp:       &ebiten.DrawTrianglesOptions{},
		scrollText:      "                HELLO, BLAH BLAH BLAH, ABCDEF GHIJKL MNOPQ RSTVU WXYZ. 01234 56789     ON ZAPPE....        ",
		sceneImage:      ebiten.NewImage(screenWidth, screenHeight),
		triangleIndices: []uint16{0, 1, 2},
	}

	if err := g.loadImages(); err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	g.init3DGeometry()
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

func (g *Game) init3DGeometry() {
	g.vertices = make([][]Vector3, 3)

	g.vertices[0] = []Vector3{
		{0, 0, 200}, {-100, 25, 100}, {100, 25, 100}, {100, -25, 100},
		{-100, -25, 100}, {200, 0, 0}, {100, 25, -100}, {100, -25, -100},
		{-200, 0, 0}, {-100, 25, -100}, {-100, -25, -100}, {0, 0, -200},
		{0, 25, 0}, {0, -25, 0},
	}

	g.vertices[1] = []Vector3{
		{0, 0, 100}, {-100, 100, 100}, {100, 100, 100}, {100, -100, 100},
		{-100, -100, 100}, {100, 0, 0}, {100, 100, -100}, {100, -100, -100},
		{-100, 0, 0}, {-100, 100, -100}, {-100, -100, -100}, {0, 0, -100},
		{0, 100, 0}, {0, -100, 0},
	}

	g.vertices[2] = []Vector3{
		{0, 0, 200}, {-100, 100, 100}, {100, 100, 100}, {100, -100, 100},
		{-100, -100, 100}, {200, 0, 0}, {100, 100, -100}, {100, -100, -100},
		{-200, 0, 0}, {-100, 100, -100}, {-100, -100, -100}, {0, 0, -200},
		{0, 200, 0}, {0, -200, 0},
	}

	g.currentVertices = make([]Vector3, 14)
	copy(g.currentVertices, g.vertices[0]) // Start with state 0
	g.transformedVerts = make([]Vector3, 14)
	g.projectedVerts = make([]projectedVertex, 14)

	greenColor := color.RGBA{R: 0x00, G: 0xaa, B: 0x00, A: 0x80} // 50% transparent (0.5 * 255 = 128 = 0x80)
	grayColor := color.RGBA{R: 0xdd, G: 0xdd, B: 0xdd, A: 0xe6}  // 90% opaque (0.9 * 255 = 230 = 0xe6)

	g.triangles = []Triangle{
		{0, 2, 1, greenColor}, {0, 3, 2, grayColor}, {0, 4, 3, greenColor}, {0, 1, 4, grayColor},
		{5, 6, 2, grayColor}, {5, 7, 6, greenColor}, {5, 3, 7, grayColor}, {5, 2, 3, greenColor},
		{8, 1, 9, grayColor}, {8, 9, 10, greenColor}, {8, 10, 4, grayColor}, {8, 4, 1, greenColor},
		{11, 6, 7, grayColor}, {11, 7, 10, greenColor}, {11, 10, 9, grayColor}, {11, 9, 6, greenColor},
		{12, 1, 2, grayColor}, {12, 2, 6, greenColor}, {12, 6, 9, grayColor}, {12, 9, 1, greenColor},
		{13, 7, 3, greenColor}, {13, 10, 7, grayColor}, {13, 4, 10, greenColor}, {13, 3, 4, grayColor},
	}
	g.trianglesDepth = make([]TriangleWithDepth, len(g.triangles))
	g.currentShape = 0 // Start with simple state 0
	g.targetShape = 1
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

	g.rotationX += 0.01
	g.rotationY += 0.02
	g.rotationZ += 0.04
	if g.rotationX >= fullRotation {
		g.rotationX -= fullRotation
	}
	if g.rotationY >= fullRotation {
		g.rotationY -= fullRotation
	}
	if g.rotationZ >= fullRotation {
		g.rotationZ -= fullRotation
	}

	g.morphTimer += 1.0
	if g.morphTimer >= 240 {
		g.morphTimer = 0
		g.currentShape = g.targetShape
		g.targetShape = (g.targetShape + 1) % 3
	}

	// Morph during the first 120 frames. The vertices already equal the target
	// at frame 120, so the hold phase needs no repeated interpolation.
	if g.morphTimer <= 120 {
		t := g.morphTimer / 120.0
		for i := range g.currentVertices {
			current := g.vertices[g.currentShape][i]
			target := g.vertices[g.targetShape][i]
			g.currentVertices[i] = Vector3{
				X: current.X + (target.X-current.X)*t,
				Y: current.Y + (target.Y-current.Y)*t,
				Z: current.Z + (target.Z-current.Z)*t,
			}
		}
	}

	// DCK owns the glyph, row-wave and column-cosine transport.
	if err := g.scrollRenderer.Update(kit.Frame{}); err != nil {
		return err
	}

	g.transform3DVertices()
	return g.starfield.Update(kit.Frame{})
}

func (g *Game) transform3DVertices() {
	cosX, sinX := math.Cos(g.rotationX), math.Sin(g.rotationX)
	cosY, sinY := math.Cos(g.rotationY), math.Sin(g.rotationY)
	cosZ, sinZ := math.Cos(g.rotationZ), math.Sin(g.rotationZ)

	for i, v := range g.currentVertices {
		x, y, z := v.X, v.Y, v.Z
		newY := y*cosX - z*sinX
		newZ := y*sinX + z*cosX
		y, z = newY, newZ
		newX := x*cosY + z*sinY
		newZ = -x*sinY + z*cosY
		x, z = newX, newZ
		newX = x*cosZ - y*sinZ
		newY = x*sinZ + y*cosZ
		x, y = newX, newY
		g.transformedVerts[i] = Vector3{X: x, Y: y, Z: z}
	}

	for i, tri := range g.triangles {
		avgZ := (g.transformedVerts[tri.V1].Z + g.transformedVerts[tri.V2].Z + g.transformedVerts[tri.V3].Z) / 3.0
		g.trianglesDepth[i] = TriangleWithDepth{tri, avgZ}
	}

	slices.SortFunc(g.trianglesDepth, func(a, b TriangleWithDepth) int {
		return cmp.Compare(a.Depth, b.Depth)
	})
}

// initScrollX initializes the scroll deformation positions
func (g *Game) draw3DObject(screen *ebiten.Image) {
	centerX := float32(screenWidth / 2)
	centerY := float32(screenHeight / 2)
	fov := 900.0
	zPos := 1000.0 // Move object further away to match original wab.com demo

	for i, v := range g.transformedVerts {
		z := v.Z + zPos
		if z <= 0 {
			z = 1
		}
		scale := float32(fov / z)
		g.projectedVerts[i].X = float32(v.X)*scale + centerX
		g.projectedVerts[i].Y = float32(v.Y)*scale + centerY
	}

	// Remove debug for clean output
	// visibleCount := 0

	for _, triDepth := range g.trianglesDepth {
		tri := triDepth.Triangle
		v0 := g.projectedVerts[tri.V1]
		v1 := g.projectedVerts[tri.V2]
		v2 := g.projectedVerts[tri.V3]

		// Enable backface culling with correct orientation
		if (v1.X-v0.X)*(v2.Y-v0.Y)-(v1.Y-v0.Y)*(v2.X-v0.X) > 0 {
			continue
		}

		// Remove debug output

		colorR := float32(tri.Color.R) / 0xff
		colorG := float32(tri.Color.G) / 0xff
		colorB := float32(tri.Color.B) / 0xff
		colorA := float32(tri.Color.A) / 0xff

		vertices := g.triangleVertices[:]
		vertices[0] = ebiten.Vertex{DstX: v0.X, DstY: v0.Y, SrcX: 0, SrcY: 0, ColorR: colorR, ColorG: colorG, ColorB: colorB, ColorA: colorA}
		vertices[1] = ebiten.Vertex{DstX: v1.X, DstY: v1.Y, SrcX: 1, SrcY: 0, ColorR: colorR, ColorG: colorG, ColorB: colorB, ColorA: colorA}
		vertices[2] = ebiten.Vertex{DstX: v2.X, DstY: v2.Y, SrcX: 0, SrcY: 1, ColorR: colorR, ColorG: colorG, ColorB: colorB, ColorA: colorA}
		// Additive blending gives the translucent green faces their Glenz look.
		if colorA < 0.9 { // For transparent green faces
			g.drawTriOp.Blend = ebiten.BlendLighter
		} else { // For opaque gray faces
			g.drawTriOp.Blend = ebiten.BlendSourceOver
		}
		g.drawTriOp.Filter = ebiten.FilterLinear
		screen.DrawTriangles(vertices, g.triangleIndices, g.colorImage, g.drawTriOp)
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	scene := g.sceneImage
	scene.Fill(color.Black)

	// Draw starfield first, but skip center area where 3D objects are
	g.starfield.Draw(scene)

	// Draw 3D objects on top
	g.draw3DObject(scene)

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
