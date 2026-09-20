// Package dma3d implements the Glenz vector demo.
package dma3d

import originalassets "dma-3d"

import (
	"bytes"
	"cmp"

	"fmt"
	"github.com/olivierh59500/democonstructionkit/composite"
	"github.com/olivierh59500/democonstructionkit/scrolling"
	"image"
	"image/color"
	_ "image/png"
	"io"
	"log"
	"math"
	"slices"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/audio"
	"github.com/olivierh59500/ym-player/pkg/stsound"

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
	verticesPerQuad    = 4
	indicesPerQuad     = 6
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
type Star struct {
	X, Y  float64
	Speed float64
	Color color.RGBA
	Size  float64
}

type YMPlayer struct {
	player *stsound.StSound
	buffer []int16
	mutex  sync.Mutex
	loop   bool
}

func NewYMPlayer(data []byte, sampleRate int, loop bool) (*YMPlayer, error) {
	player := stsound.CreateWithRate(sampleRate)
	if err := player.LoadMemory(data); err != nil {
		player.Destroy()
		return nil, fmt.Errorf("failed to load YM data: %w", err)
	}
	player.SetLoopMode(loop)
	return &YMPlayer{
		player: player,
		buffer: make([]int16, 4096),
		loop:   loop,
	}, nil
}

func (y *YMPlayer) Read(p []byte) (n int, err error) {
	y.mutex.Lock()
	defer y.mutex.Unlock()

	if y.player == nil {
		return 0, io.EOF
	}
	if len(p) > 0 && len(p) < pcmFrameBytes {
		return 0, io.ErrShortBuffer
	}

	samplesNeeded := len(p) / pcmFrameBytes
	processed := 0
	for processed < samplesNeeded {
		chunkSize := samplesNeeded - processed
		if chunkSize > len(y.buffer) {
			chunkSize = len(y.buffer)
		}
		if !y.player.Compute(y.buffer[:chunkSize], chunkSize) {
			if !y.loop {
				clear(p[processed*pcmFrameBytes : samplesNeeded*pcmFrameBytes])
				err = io.EOF
				break
			}
		}
		for i := 0; i < chunkSize; i++ {
			sample := y.buffer[i]
			byteOffset := (processed + i) * pcmFrameBytes
			p[byteOffset] = byte(sample)
			p[byteOffset+1] = byte(sample >> 8)
			p[byteOffset+2] = byte(sample)
			p[byteOffset+3] = byte(sample >> 8)
		}
		processed += chunkSize
	}
	return samplesNeeded * pcmFrameBytes, err
}

func (y *YMPlayer) Close() error {
	y.mutex.Lock()
	defer y.mutex.Unlock()
	if y.player != nil {
		y.player.Destroy()
		y.player = nil
	}
	return nil
}

type Game struct {
	scrollRenderer     *scrolling.Scrolling
	stripBatch         *composite.QuadBatch
	fontImg            *ebiten.Image
	colorImage         *ebiten.Image
	audioContext       *audio.Context
	audioPlayer        *audio.Player
	ymPlayer           *YMPlayer
	vertices           [][]Vector3
	currentVertices    []Vector3
	triangles          []Triangle
	transformedVerts   []Vector3
	trianglesDepth     []TriangleWithDepth
	stars              []Star
	frame              int
	rotationX          float64
	rotationY          float64
	rotationZ          float64
	morphTimer         float64
	currentShape       int
	targetShape        int
	scrollText         string
	scrollGlyphs       []int
	scrollTextWidth    float64
	scrollPos          float64
	scrollXData        []float64
	scrollOffset       float64
	fontGlyphs         [fontGlyphCount]*ebiten.Image
	scrollLineVertices []ebiten.Vertex
	scrollLineIndices  []uint16
	scrollColVertices  []ebiten.Vertex
	scrollColIndices   []uint16
	drawTriOp          *ebiten.DrawTrianglesOptions
	audioInitAttempted bool
	sceneImage         *ebiten.Image
	scrollWorkBuffer   *ebiten.Image
	scrollDeformBuffer *ebiten.Image
	projectedVerts     []projectedVertex
	triangleVertices   [3]ebiten.Vertex
	triangleIndices    []uint16
	starVertices       []ebiten.Vertex
	starIndices        []uint16
}

type projectedVertex struct {
	X, Y float32
}

func NewGame() (*Game, error) {
	g := &Game{
		drawTriOp:          &ebiten.DrawTrianglesOptions{},
		scrollText:         "                HELLO, BLAH BLAH BLAH, ABCDEF GHIJKL MNOPQ RSTVU WXYZ. 01234 56789     ON ZAPPE....        ",
		sceneImage:         ebiten.NewImage(screenWidth, screenHeight),
		scrollWorkBuffer:   ebiten.NewImage(screenWidth+scrollWorkPadding, glyphHeight),
		scrollDeformBuffer: ebiten.NewImage(screenWidth, glyphHeight),
		triangleIndices:    []uint16{0, 1, 2},
	}

	if err := g.loadImages(); err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	g.init3DGeometry()
	g.initStarfield()
	g.initScrollX()
	g.initScrollText()
	g.scrollLineVertices, g.scrollLineIndices = newQuadBatch(scrollLineCount)
	g.scrollColVertices, g.scrollColIndices = newQuadBatch(scrollColumnCount)
	g.colorImage = ebiten.NewImage(1, 1)
	g.colorImage.Fill(color.RGBA{255, 255, 255, 255}) // White with full alpha

	return g, nil
}

func (g *Game) initAudio() error {
	g.audioContext = audio.NewContext(sampleRate)
	var err error
	g.ymPlayer, err = NewYMPlayer(musicData, sampleRate, true)
	if err != nil {
		return err
	}
	g.audioPlayer, err = g.audioContext.NewPlayer(g.ymPlayer)
	if err != nil {
		if closeErr := g.ymPlayer.Close(); closeErr != nil {
			log.Printf("close YM player after audio initialization failure: %v", closeErr)
		}
		g.ymPlayer = nil
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
	for index := range g.fontGlyphs {
		row, col := index/10, index%10
		g.fontGlyphs[index] = g.fontImg.SubImage(image.Rect(
			col*glyphWidth,
			row*glyphHeight,
			(col+1)*glyphWidth,
			(row+1)*glyphHeight,
		)).(*ebiten.Image)
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

func (g *Game) initStarfield() {
	starParams := []struct {
		count int
		speed float64
		color color.RGBA
		size  float64
	}{
		{35, 11.2, color.RGBA{0xE0, 0xA0, 0xA0, 0xFF}, 2},
		{35, 5.6, color.RGBA{0xC0, 0x60, 0x60, 0xFF}, 2},
		{35, 2.8, color.RGBA{0x80, 0x40, 0x40, 0xFF}, 2},
	}
	starCount := 0
	for _, param := range starParams {
		starCount += param.count
	}
	g.stars = make([]Star, 0, starCount)

	for _, param := range starParams {
		for i := 0; i < param.count; i++ {
			star := Star{
				X:     math.Mod(float64(i*73+int(param.speed)*137)*1.234/1000.0*float64(screenWidth), float64(screenWidth)),
				Y:     math.Mod(float64(i*97+int(param.speed)*211)*2.345/1000.0*280, 280), // Stars only in upper part
				Speed: param.speed,
				Color: param.color,
				Size:  param.size,
			}
			g.stars = append(g.stars, star)
		}
	}
	g.starVertices = make([]ebiten.Vertex, 0, len(g.stars)*verticesPerQuad)
	g.starIndices = make([]uint16, 0, len(g.stars)*indicesPerQuad)
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

	g.frame++
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

	// Update TCB-style scroll
	g.scrollPos -= 4.0
	if g.scrollPos < -g.scrollTextWidth {
		g.scrollPos = float64(screenWidth)
	}
	g.scrollOffset += 0.1
	if g.scrollOffset >= fullRotation {
		g.scrollOffset -= fullRotation
	}

	g.transform3DVertices()
	g.updateStarfield()
	return nil
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

func (g *Game) updateStarfield() {
	for i := range g.stars {
		// Move stars from left to right
		g.stars[i].X += g.stars[i].Speed

		// Reset star when it goes off the right side
		if g.stars[i].X > float64(screenWidth) {
			g.stars[i].X -= float64(screenWidth)
			g.stars[i].Y = math.Mod(float64(i*97+g.frame), 280)
		}
	}
}

func (g *Game) drawStarfieldWithMask(screen *ebiten.Image) {
	centerX := screenWidth / 2
	centerY := screenHeight / 2
	vertices := g.starVertices[:0]
	indices := g.starIndices[:0]

	for _, star := range g.stars {
		x, y := int(star.X), int(star.Y)

		// Skip stars in the center area where 3D objects appear (roughly 400x300 pixels centered)
		if x >= centerX-200 && x <= centerX+200 && y >= centerY-150 && y <= centerY+150 {
			continue
		}

		base := uint16(len(vertices))
		x0, y0 := float32(star.X), float32(star.Y)
		x1, y1 := x0+float32(star.Size), y0+float32(star.Size)
		r := float32(star.Color.R) / 0xff
		green := float32(star.Color.G) / 0xff
		b := float32(star.Color.B) / 0xff
		a := float32(star.Color.A) / 0xff
		vertices = append(vertices,
			ebiten.Vertex{DstX: x0, DstY: y0, ColorR: r, ColorG: green, ColorB: b, ColorA: a},
			ebiten.Vertex{DstX: x1, DstY: y0, ColorR: r, ColorG: green, ColorB: b, ColorA: a},
			ebiten.Vertex{DstX: x0, DstY: y1, ColorR: r, ColorG: green, ColorB: b, ColorA: a},
			ebiten.Vertex{DstX: x1, DstY: y1, ColorR: r, ColorG: green, ColorB: b, ColorA: a},
		)
		indices = append(indices, base, base+1, base+2, base+1, base+3, base+2)
	}

	g.starVertices = vertices
	g.starIndices = indices
	if len(indices) > 0 {
		screen.DrawTriangles(vertices, indices, g.colorImage, nil)
	}
}

// initScrollX initializes the scroll deformation positions
func (g *Game) initScrollX() {
	g.scrollXData = make([]float64, 0, scrollWaveDataSize)

	// First wave pattern
	stp1 := 7.0 / 180.0 * math.Pi
	stp2 := 3.0 / 180.0 * math.Pi
	for i := 0; i < 389; i++ {
		x := 20*math.Sin(float64(i)*stp1) + 30*math.Cos(float64(i)*stp2)
		g.scrollXData = append(g.scrollXData, x)
	}

	// Second wave pattern
	stp1 = 72.0 / 180.0 * math.Pi
	for i := 0; i < 120; i++ {
		x := 4 * math.Sin(float64(i)*stp1)
		g.scrollXData = append(g.scrollXData, x)
	}

	// Third wave pattern
	stp1 = 8.0 / 180.0 * math.Pi
	for i := 0; i < 68; i++ {
		x := 40 * math.Sin(float64(i)*stp1)
		g.scrollXData = append(g.scrollXData, x)
	}

	// Repeat first pattern
	stp1 = 7.0 / 180.0 * math.Pi
	stp2 = 3.0 / 180.0 * math.Pi
	for i := 0; i < 389; i++ {
		x := 20*math.Sin(float64(i)*stp1) + 30*math.Cos(float64(i)*stp2)
		g.scrollXData = append(g.scrollXData, x)
	}

	// Small wave
	stp1 = 72.0 / 180.0 * math.Pi
	for i := 0; i < 36; i++ {
		x := 4 * math.Sin(float64(i)*stp1)
		g.scrollXData = append(g.scrollXData, x)
	}

	// Final wave
	stp1 = 8.0 / 180.0 * math.Pi
	for i := 0; i < 189; i++ {
		x := 30 * math.Sin(float64(i)*stp1)
		g.scrollXData = append(g.scrollXData, x)
	}
}

func (g *Game) initScrollText() {
	g.scrollGlyphs = make([]int, 0, len(g.scrollText))
	for _, ch := range g.scrollText {
		index, found := charToFontIndex(ch)
		if !found {
			index = -1
		}
		g.scrollGlyphs = append(g.scrollGlyphs, index)
	}
	g.scrollTextWidth = float64(len(g.scrollGlyphs) * glyphWidth)
}

func newQuadBatch(quadCount int) ([]ebiten.Vertex, []uint16) {
	vertices := make([]ebiten.Vertex, quadCount*verticesPerQuad)
	indices := make([]uint16, quadCount*indicesPerQuad)
	for quad := 0; quad < quadCount; quad++ {
		vertexBase := quad * verticesPerQuad
		for i := 0; i < verticesPerQuad; i++ {
			vertices[vertexBase+i].ColorR = 1
			vertices[vertexBase+i].ColorG = 1
			vertices[vertexBase+i].ColorB = 1
			vertices[vertexBase+i].ColorA = 1
		}

		indexBase := quad * indicesPerQuad
		base := uint16(vertexBase)
		indices[indexBase] = base
		indices[indexBase+1] = base + 1
		indices[indexBase+2] = base + 2
		indices[indexBase+3] = base + 1
		indices[indexBase+4] = base + 3
		indices[indexBase+5] = base + 2
	}
	return vertices, indices
}

// charToFontIndex converts a character to its position in the font bitmap
func charToFontIndex(ch rune) (int, bool) {
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

func (g *Game) drawScrollText(screen *ebiten.Image) {
	workBuffer, deformBuffer := g.scrollWorkBuffer, g.scrollDeformBuffer
	workBuffer.Clear()
	deformBuffer.Clear()
	if g.scrollRenderer == nil {
		images := make([]*ebiten.Image, len(g.scrollGlyphs))
		for i, index := range g.scrollGlyphs {
			if index >= 0 {
				images[i] = g.fontGlyphs[index]
			}
		}
		var err error
		g.scrollRenderer, err = scrolling.FromImages(images, glyphWidth)
		if err != nil {
			panic(err)
		}
		g.stripBatch = composite.NewQuadBatch(max(scrollLineCount, scrollColumnCount))
		g.stripBatch.AlternateDiagonal = true
	}
	first := max(0, int(math.Floor((-float64(glyphWidth)-g.scrollPos)/glyphWidth))+1)
	last := min(len(g.scrollGlyphs), int(math.Ceil((float64(workBuffer.Bounds().Dx())-g.scrollPos)/glyphWidth)))
	state := scrolling.IdentityState()
	state.X = g.scrollPos
	state.First = first
	state.End = last
	g.scrollRenderer.DrawAt(workBuffer, state)
	g.stripBatch.Begin(deformBuffer, workBuffer)
	for line := 0; line < scrollLineCount; line++ {
		x := int(g.scrollXData[(g.frame+line)%len(g.scrollXData)] + glyphWidth)
		y := line * scrollLineHeight
		g.stripBatch.Rect(image.Rect(x, y, x+screenWidth, y+scrollLineHeight), 0, float32(y), screenWidth, scrollLineHeight)
	}
	g.stripBatch.Flush()
	g.stripBatch.Begin(screen, deformBuffer)
	for col := 0; col < scrollColumnCount; col++ {
		x := col * scrollColumnWidth
		y := float32(380 + 35 + math.Cos(g.scrollOffset+float64(col)*.1)*35)
		g.stripBatch.Rect(image.Rect(x, 0, x+scrollColumnWidth, glyphHeight), float32(x), y, scrollColumnWidth, glyphHeight)
	}
	g.stripBatch.Flush()
}

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
	g.drawStarfieldWithMask(scene)

	// Draw 3D objects on top
	g.draw3DObject(scene)

	// Draw scrolling text
	g.drawScrollText(scene)

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
	if g.audioPlayer != nil {
		if err := g.audioPlayer.Close(); err != nil {
			log.Printf("close audio player: %v", err)
		}
	}
	if g.ymPlayer != nil {
		if err := g.ymPlayer.Close(); err != nil {
			log.Printf("close YM player: %v", err)
		}
	}
}
