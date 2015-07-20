http://en.wikibooks.org/wiki/OpenGL_Programming/Scientific_OpenGL_Tutorial_03

vertex:

attribute vec2 coord2d;
uniform mat4 transform;
 
void main(void) {
  gl_Position = transform * vec4(coord2d.xy, 0, 1);
}

frag:
uniform vec4 color;
 
void main(void) {
  gl_FragColor = color;
}


GLint uniform_transform = glGetUniformLocation(program, "transform");
 
glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(scale_x, 1, 1)), glm::vec3(offset_x, 0, 0));
glUniformMatrix4fv(uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));


const int margin = 20;
const int ticksize = 10;


int window_width = glutGet(GLUT_WINDOW_WIDTH);
int window_height = glutGet(GLUT_WINDOW_HEIGHT);
 
glViewport(
  margin + ticksize,
  margin + ticksize,
  window_width - margin * 2 - ticksize,
  window_height - margin * 2 - ticksize
);


glScissor(
  margin + ticksize,
  margin + ticksize,
  window_width - margin * 2 - ticksize,
  window_height - margin * 2 - ticksize
);
 
glEnable(GL_SCISSOR_TEST);





glViewport(0, 0, window_width, window_height);
glDisable(GL_SCISSOR_TEST);


glm::mat4 viewport_transform(float x, float y, float width, float height) {
  // Calculate how to translate the x and y coordinates:
  float offset_x = (2.0 * x + (width - window_width)) / window_width;
  float offset_y = (2.0 * y + (height - window_height)) / window_height;
 
  // Calculate how to rescale the x and y coordinates:
  float scale_x = width / window_width;
  float scale_y = height / window_height;
 
  return glm::scale(glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));
}






transform = viewport_transform(
  margin + ticksize,
  margin + ticksize,
  window_width - margin * 2 - ticksize,
  window_height - margin * 2 - ticksize,
);
 
glUniformMatrix4fv(uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));

Then we draw our box, in black:

GLuint box_vbo;
glGenBuffers(1, &box_vbo);
glBindBuffer(GL_ARRAY_BUFFER, box_vbo);
 
static const point box[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
glBufferData(GL_ARRAY_BUFFER, sizeof box, box, GL_STATIC_DRAW);
 
GLfloat black[4] = {0, 0, 0, 1};
glUniform4fv(uniform_color, 1, black);
 
glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
glDrawArrays(GL_LINE_LOOP, 0, 4);





float pixel_x = 2.0 / (window_width - border * 2 - ticksize);
float pixel_y = 2.0 / (window_height - border * 2 - ticksize);

Now that we know that, we can calculate the coordinates of the 42 vertices we need to draw 21 tick marks, and put those in a VBO:

GLuint ticks_vbo;
glGenBuffers(1, &ticks_vbo);
glBindBuffer(GL_ARRAY_BUFFER, ticks_vbo);
 
point ticks[42];
 
for(int i = 0; i <= 20; i++) {
  float y = -1 + i * 0.1;
  ticks[i * 2].x = -1;
  ticks[i * 2].y = y; 
  ticks[i * 2 + 1].x = -1 - ticksize * pixel_x;
  ticks[i * 2 + 1].y = y; 
}
 
glBufferData(GL_ARRAY_BUFFER, sizeof ticks, ticks, GL_STREAM_DRAW);
 
glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
glDrawArrays(GL_LINES, 0, 42);



float tickscale = (i % 10) ? 0.5 : 1; 
ticks[i * 2 + 1].x = -1 - ticksize * tickscale * pixel_x;


float tickspacing = 0.1 * powf(10, -floor(log10(scale_x)));

float left = -1.0 / scale_x - offset_x;
float right = 1.0 / scale_x - offset_x;


int left_i = ceil(left / tickspacing);
int right_i = floor(right / tickspacing);




float rem = left_i * tickspacing - left;

Now we can calculate the coordinate of the left most tick mark in the coordinate system we are going to draw with:

float firsttick = -1.0 + rem * scale_x;



int nticks = right_i - left_i + 1;
if(nticks > 21)
  nticks = 21;
 
for(int i = 0; i < nticks; i++) {
  float x = firsttick + i * tickspacing * scale_x;
  float tickscale = ((i + left_i) % 10) ? 0.5 : 1; 
  ticks[i * 2].x = x; 
  ticks[i * 2].y = -1;
  ticks[i * 2 + 1].x = x; 
  ticks[i * 2 + 1].y = -1 - ticksize * tickscale * pixel_y;
}
 
glBufferData(GL_ARRAY_BUFFER, nticks * sizeof *ticks, ticks, GL_STREAM_DRAW);
 
glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
glDrawArrays(GL_LINES, 0, nticks * 2);
