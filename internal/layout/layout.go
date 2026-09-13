// Package layout contains display-size calculations that do not depend on the
// graphics backend.
package layout

const (
	SceneWidth  = 640
	SceneHeight = 480
	MaxWidth    = 1280
)

// LogicalWidth keeps the scene's height and expands the logical surface to
// match the device aspect ratio. Narrow displays retain the full scene.
func LogicalWidth(outsideWidth, outsideHeight int) int {
	if outsideWidth <= 0 || outsideHeight <= 0 {
		return SceneWidth
	}

	width := (outsideWidth*SceneHeight + outsideHeight - 1) / outsideHeight
	if width < SceneWidth {
		return SceneWidth
	}
	if width > MaxWidth {
		return MaxWidth
	}
	return width
}
