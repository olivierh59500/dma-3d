
// Glenz Vectors Demo - Go/Ebiten/ym-player demo
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"image"
	"image/color"
	_ "image/png"
	"io"
	"log"
	"math"
	"sort"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/audio"
	"github.com/olivierh59500/ym-player/pkg/stsound"
)

const (
	screenWidth   = 640
	screenHeight  = 480
	sampleRate    = 44100
	fontCharWidth = 32
	fontCharHeight= 32
	waveIncrement = 0.2
)

var (
	//go:embed assets/tcb_rep_font.png
	fontData []byte
	//go:embed assets/music.ym
	musicData []byte
)

type Vector3 struct{ X, Y, Z float64 }
type Triangle struct {
	V1, V2, V3 int
	Color      color.Color
	Alpha      float32
}
type TriangleWithDepth struct {
	Triangle
	Depth float64
}
type Star struct {
	X, Y     float64
	Speed    float64
	Color    color.Color
	Size     float64
}

type YMPlayer struct {
	player       *stsound.StSound
	sampleRate   int
	buffer       []int16
	mutex        sync.Mutex
	position     int64
	totalSamples int64
	loop         bool
	volume       float64
}

func NewYMPlayer(data []byte, sampleRate int, loop bool) (*YMPlayer, error) {
	player := stsound.CreateWithRate(sampleRate)
	if err := player.LoadMemory(data); err != nil {
		player.Destroy()
		return nil, fmt.Errorf("failed to load YM data: %w", err)
	}
	player.SetLoopMode(loop)
	info := player.GetInfo()
	totalSamples := int64(info.MusicTimeInMs) * int64(sampleRate) / 1000
	return &YMPlayer{
		player:       player,
		sampleRate:   sampleRate,
		buffer:       make([]int16, 4096),
		totalSamples: totalSamples,
		loop:         loop,
		volume:       1.0,
	}, nil
}

func (y *YMPlayer) Read(p []byte) (n int, err error) {
	y.mutex.Lock()
	defer y.mutex.Unlock()
	samplesNeeded := len(p) / 4
	outBuffer := make([]int16, samplesNeeded*2)
	processed := 0
	for processed < samplesNeeded {
		chunkSize := samplesNeeded - processed
		if chunkSize > len(y.buffer) {
			chunkSize = len(y.buffer)
		}
		if !y.player.Compute(y.buffer[:chunkSize], chunkSize) {
			if !y.loop {
				for i := processed * 2; i < len(outBuffer); i++ {
					outBuffer[i] = 0
				}
				err = io.EOF
				break
			}
		}
		for i := 0; i < chunkSize; i++ {
			sample := int16(float64(y.buffer[i]) * y.volume)
			outBuffer[(processed+i)*2] = sample
			outBuffer[(processed+i)*2+1] = sample
		}
		processed += chunkSize
		y.position += int64(chunkSize)
	}
	buf := make([]byte, 0, len(outBuffer)*2)
	for _, sample := range outBuffer {
		buf = append(buf, byte(sample), byte(sample>>8))
	}
	copy(p, buf)
	n = len(buf)
	if n > len(p) {
		n = len(p)
	}
	return n, err
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
	fontImg          *ebiten.Image
	colorImage       *ebiten.Image
	audioContext     *audio.Context
	audioPlayer      *audio.Player
	ymPlayer         *YMPlayer
	vertices         [][]Vector3
	currentVertices  []Vector3
	triangles        []Triangle
	transformedVerts []Vector3
	trianglesDepth   []TriangleWithDepth
	stars            []Star
	frame            int
	rotationX        float64
	rotationY        float64
	rotationZ        float64
	morphTimer       float64
	currentShape     int
	targetShape      int
	scrollText       string
	scrollPos        float64
	scrollXData      []float64
	scrollOffset     float64
	drawOp           *ebiten.DrawImageOptions
	drawTriOp        *ebiten.DrawTrianglesOptions
}

func NewGame() (*Game, error) {
	g := &Game{
		drawOp:     &ebiten.DrawImageOptions{},
		drawTriOp:  &ebiten.DrawTrianglesOptions{},
		scrollText: "                HELLO, BLAH BLAH BLAH, ABCDEF GHIJKL MNOPQ RSTVU WXYZ. 01234 56789     ON ZAPPE....        ",
	}

	if err := g.initAudio(); err != nil {
		return nil, fmt.Errorf("failed to initialize audio: %w", err)
	}
	if err := g.loadImages(); err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	g.init3DGeometry()
	g.initStarfield()
	g.initScrollX()
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
		g.ymPlayer.Close()
		return err
	}
	g.audioPlayer.SetVolume(0.7)
	g.audioPlayer.Play()
	return nil
}

func (g *Game) loadImages() error {
	var err error

	img, _, err := image.Decode(bytes.NewReader(fontData))
	if err != nil { return fmt.Errorf("failed to load TCB font: %w", err) }
	g.fontImg = ebiten.NewImageFromImage(img)

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

	greenColor := color.RGBA{R: 0x00, G: 0xaa, B: 0x00, A: 0x80} // 50% transparent (0.5 * 255 = 128 = 0x80)
	grayColor := color.RGBA{R: 0xdd, G: 0xdd, B: 0xdd, A: 0xe6}  // 90% opaque (0.9 * 255 = 230 = 0xe6)

	g.triangles = []Triangle{
		{0, 2, 1, greenColor, 1.0}, {0, 3, 2, grayColor, 1.0}, {0, 4, 3, greenColor, 1.0}, {0, 1, 4, grayColor, 1.0},
		{5, 6, 2, grayColor, 1.0}, {5, 7, 6, greenColor, 1.0}, {5, 3, 7, grayColor, 1.0}, {5, 2, 3, greenColor, 1.0},
		{8, 1, 9, grayColor, 1.0}, {8, 9, 10, greenColor, 1.0}, {8, 10, 4, grayColor, 1.0}, {8, 4, 1, greenColor, 1.0},
		{11, 6, 7, grayColor, 1.0}, {11, 7, 10, greenColor, 1.0}, {11, 10, 9, grayColor, 1.0}, {11, 9, 6, greenColor, 1.0},
		{12, 1, 2, grayColor, 1.0}, {12, 2, 6, greenColor, 1.0}, {12, 6, 9, grayColor, 1.0}, {12, 9, 1, greenColor, 1.0},
		{13, 7, 3, greenColor, 1.0}, {13, 10, 7, grayColor, 1.0}, {13, 4, 10, greenColor, 1.0}, {13, 3, 4, grayColor, 1.0},
	}
	g.trianglesDepth = make([]TriangleWithDepth, len(g.triangles))
	g.currentShape = 0  // Start with simple state 0
	g.targetShape = 1
}

func (g *Game) initStarfield() {
	starParams := []struct {
		count int
		speed float64
		color color.Color
		size  float64
	}{
		{35, 11.2, color.RGBA{0xE0, 0xA0, 0xA0, 0xFF}, 2},
		{35, 5.6, color.RGBA{0xC0, 0x60, 0x60, 0xFF}, 2},
		{35, 2.8, color.RGBA{0x80, 0x40, 0x40, 0xFF}, 2},
	}

	for _, param := range starParams {
		for i := 0; i < param.count; i++ {
			star := Star{
				X:     math.Mod(float64(i*73+int(param.speed)*137) * 1.234 / 1000.0 * float64(screenWidth), float64(screenWidth)),
				Y:     math.Mod(float64(i*97+int(param.speed)*211) * 2.345 / 1000.0 * 280, 280), // Stars only in upper part
				Speed: param.speed,
				Color: param.color,
				Size:  param.size,
			}
			g.stars = append(g.stars, star)
		}
	}
}

func (g *Game) Update() error {
	g.frame++
	g.rotationX += 0.01
	g.rotationY += 0.02
	g.rotationZ += 0.04

	g.morphTimer += 1.0
	if g.morphTimer >= 240 {
		g.morphTimer = 0
		g.currentShape = g.targetShape
		g.targetShape = (g.targetShape + 1) % 3
	}

	// Morph during first 120 frames, hold during last 120 frames
	var t float64
	if g.morphTimer <= 120 {
		t = g.morphTimer / 120.0
	} else {
		t = 1.0
	}
	for i := 0; i < 14; i++ {
		current := g.vertices[g.currentShape][i]
		target := g.vertices[g.targetShape][i]
		g.currentVertices[i] = Vector3{
			X: current.X + (target.X-current.X)*t,
			Y: current.Y + (target.Y-current.Y)*t,
			Z: current.Z + (target.Z-current.Z)*t,
		}
	}

	// Update TCB-style scroll
	g.scrollPos -= 4.0
	textWidth := float64(len(g.scrollText) * 64) // 64 = char width
	if g.scrollPos < -textWidth {
		g.scrollPos = float64(screenWidth)
	}
	g.scrollOffset += 0.1

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

	sort.Slice(g.trianglesDepth, func(i, j int) bool {
		return g.trianglesDepth[i].Depth < g.trianglesDepth[j].Depth
	})
}

func (g *Game) updateStarfield() {
	for i := range g.stars {
		// Move stars from left to right
		g.stars[i].X += g.stars[i].Speed
		
		// Reset star when it goes off the right side
		if g.stars[i].X > float64(screenWidth) {
			g.stars[i].X = 0
			g.stars[i].Y = math.Mod(float64(i*97+g.frame), 280)
		}
	}
}

func (g *Game) drawStarfieldWithMask(screen *ebiten.Image) {
	centerX := screenWidth / 2
	centerY := screenHeight / 2
	
	for _, star := range g.stars {
		x := int(star.X)
		y := int(star.Y)
		
		// Skip stars outside screen bounds
		if x < 0 || x >= screenWidth || y < 0 || y >= screenHeight {
			continue
		}
		
		// Skip stars in the center area where 3D objects appear (roughly 400x300 pixels centered)
		if x >= centerX-200 && x <= centerX+200 && y >= centerY-150 && y <= centerY+150 {
			continue
		}
		
		// Draw star as filled rectangle
		for dy := 0; dy < int(star.Size); dy++ {
			for dx := 0; dx < int(star.Size); dx++ {
				if x+dx < screenWidth && y+dy < screenHeight {
					screen.Set(x+dx, y+dy, star.Color)
				}
			}
		}
	}
}

// initScrollX initializes the scroll deformation positions
func (g *Game) initScrollX() {
	g.scrollXData = make([]float64, 0)

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

// charToFontIndex converts a character to its position in the font bitmap
func charToFontIndex(ch rune) (int, bool) {
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
	case '0':
		return 16, true
	case '1':
		return 17, true
	case '2':
		return 18, true
	case '3':
		return 19, true
	case '4':
		return 20, true
	case '5':
		return 21, true
	case '6':
		return 22, true
	case '7':
		return 23, true
	case '8':
		return 24, true
	case '9':
		return 25, true
	case ':':
		return 27, true
	case ';':
		return 28, true
	case '?':
		return 31, true
	case 'A':
		return 33, true
	case 'B':
		return 34, true
	case 'C':
		return 35, true
	case 'D':
		return 36, true
	case 'E':
		return 37, true
	case 'F':
		return 38, true
	case 'G':
		return 39, true
	case 'H':
		return 40, true
	case 'I':
		return 41, true
	case 'J':
		return 42, true
	case 'K':
		return 43, true
	case 'L':
		return 44, true
	case 'M':
		return 45, true
	case 'N':
		return 46, true
	case 'O':
		return 47, true
	case 'P':
		return 48, true
	case 'Q':
		return 49, true
	case 'R':
		return 50, true
	case 'S':
		return 51, true
	case 'T':
		return 52, true
	case 'U':
		return 53, true
	case 'V':
		return 54, true
	case 'W':
		return 55, true
	case 'X':
		return 56, true
	case 'Y':
		return 57, true
	case 'Z':
		return 58, true
	default:
		return 0, false
	}
}

func (g *Game) drawScrollText(screen *ebiten.Image) {
	// Create work buffer for TCB-style scroll deformation
	workBuffer := ebiten.NewImage(screenWidth+512, 50)
	deformBuffer := ebiten.NewImage(screenWidth, 50)
	
	// Draw text to work buffer
	x := g.scrollPos
	for _, ch := range g.scrollText {
		if ch == ' ' {
			x += 64 // char width
			continue
		}

		// Get character position in font
		charIndex, found := charToFontIndex(ch)
		if !found {
			x += 64
			continue
		}

		row := charIndex / 10
		col := charIndex % 10

		sx := col * 64 // char width
		sy := row * 50 // char height

		if x > -64 && x < float64(workBuffer.Bounds().Dx()) {
			op := &ebiten.DrawImageOptions{}
			op.GeoM.Translate(x, 0)

			subImg := g.fontImg.SubImage(
				image.Rect(sx, sy, sx+64, sy+50),
			).(*ebiten.Image)

			workBuffer.DrawImage(subImg, op)
		}

		x += 64
	}

	// Apply deformation line by line
	for y := 0; y < 25; y++ {
		offsetX := g.scrollXData[(g.frame+y)%len(g.scrollXData)] + 64

		// Draw each line with horizontal offset
		srcRect := image.Rect(int(offsetX), y*2, int(offsetX)+screenWidth, (y+1)*2)
		if srcRect.Min.X < 0 {
			srcRect.Min.X = 0
		}
		if srcRect.Max.X > workBuffer.Bounds().Dx() {
			srcRect.Max.X = workBuffer.Bounds().Dx()
		}

		subImg := workBuffer.SubImage(srcRect).(*ebiten.Image)

		dstOp := &ebiten.DrawImageOptions{}
		dstOp.GeoM.Translate(0, float64(y*2))
		deformBuffer.DrawImage(subImg, dstOp)
	}

	// Draw deformed scroll with vertical wave
	for x := 0; x < 40; x++ {
		yOffset := 35 + math.Cos(g.scrollOffset+float64(x)*0.1)*35

		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(float64(x*16), 380+yOffset)

		subImg := deformBuffer.SubImage(
			image.Rect(x*16, 0, (x+1)*16, 50),
		).(*ebiten.Image)

		screen.DrawImage(subImg, op)
	}
}

func (g *Game) draw3DObject(screen *ebiten.Image) {
	centerX := float32(screenWidth / 2)
	centerY := float32(screenHeight / 2)
	fov := 900.0
	zPos := 1000.0 // Move object further away to match original wab.com demo

	projectedVerts := make([]struct{ X, Y float32 }, len(g.transformedVerts))
	for i, v := range g.transformedVerts {
		z := v.Z + zPos
		if z <= 0 { z = 1 }
		scale := float32(fov / z)
		projectedVerts[i].X = float32(v.X)*scale + centerX
		projectedVerts[i].Y = float32(v.Y)*scale + centerY
	}

	// Remove debug for clean output
	// visibleCount := 0

	for _, triDepth := range g.trianglesDepth {
		tri := triDepth.Triangle
		v0 := projectedVerts[tri.V1]
		v1 := projectedVerts[tri.V2]
		v2 := projectedVerts[tri.V3]

		// Enable backface culling with correct orientation
		if (v1.X-v0.X)*(v2.Y-v0.Y)-(v1.Y-v0.Y)*(v2.X-v0.X) > 0 {
			continue
		}

		// Remove debug output

		r, gr, b, a := tri.Color.RGBA()
		colorR := float32(r) / 65535.0
		colorG := float32(gr) / 65535.0
		colorB := float32(b) / 65535.0
		colorA := float32(a) / 65535.0 // Use color's alpha, not separate alpha

		vertices := []ebiten.Vertex{
			{DstX: v0.X, DstY: v0.Y, SrcX: 0, SrcY: 0, ColorR: colorR, ColorG: colorG, ColorB: colorB, ColorA: colorA},
			{DstX: v1.X, DstY: v1.Y, SrcX: 1, SrcY: 0, ColorR: colorR, ColorG: colorG, ColorB: colorB, ColorA: colorA},
			{DstX: v2.X, DstY: v2.Y, SrcX: 0, SrcY: 1, ColorR: colorR, ColorG: colorG, ColorB: colorB, ColorA: colorA},
		}
		indices := []uint16{0, 1, 2}
		// Try different blend mode for true transparency  
		if colorA < 0.9 { // For transparent green faces
			g.drawTriOp.Blend = ebiten.BlendLighter
		} else { // For opaque gray faces
			g.drawTriOp.Blend = ebiten.BlendSourceOver
		}
		g.drawTriOp.Filter = ebiten.FilterLinear
		screen.DrawTriangles(vertices, indices, g.colorImage, g.drawTriOp)
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.Black)
	
	// Draw starfield first, but skip center area where 3D objects are
	g.drawStarfieldWithMask(screen)
	
	// Draw 3D objects on top
	g.draw3DObject(screen)
	
	// Draw scrolling text
	g.drawScrollText(screen)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func (g *Game) Cleanup() {
	if g.audioPlayer != nil {
		g.audioPlayer.Close()
	}
	if g.ymPlayer != nil {
		g.ymPlayer.Close()
	}
}

func main() {
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("DMA 3d Demo (Go/Ebiten/ym-player)")
	game, err := NewGame()
	if err != nil {
		log.Fatal(err)
	}
	defer game.Cleanup()
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
