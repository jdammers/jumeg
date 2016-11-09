attribute vec2 pos2d;
uniform mat4 trafo_matrix;
uniform vec4 color;

varying vec4 frg_color;

void main(void) {
	gl_Position = trafo_matrix * vec4(pos2d,0.0, 1.0);
	frg_color = color;
}

