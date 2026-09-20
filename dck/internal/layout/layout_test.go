package layout

import "testing"

func TestLogicalWidth(t *testing.T) {
	tests := []struct {
		name                 string
		outsideWidth, height int
		want                 int
	}{
		{name: "invalid", want: SceneWidth},
		{name: "desktop 4:3", outsideWidth: 640, height: 480, want: 640},
		{name: "portrait clamps to scene", outsideWidth: 480, height: 640, want: 640},
		{name: "Pixel 10a landscape", outsideWidth: 2424, height: 1080, want: 1078},
		{name: "ultrawide cap", outsideWidth: 4000, height: 480, want: MaxWidth},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LogicalWidth(tt.outsideWidth, tt.height); got != tt.want {
				t.Fatalf("LogicalWidth(%d, %d) = %d; want %d", tt.outsideWidth, tt.height, got, tt.want)
			}
		})
	}
}
