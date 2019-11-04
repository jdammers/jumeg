#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Very simple transformation library that is needed for some examples.

http://www.labri.fr/perso/nrougier/teaching/opengl/#hello-flat-world
"""

import math
import numpy
import numpy as np

'''
glm::mat4 viewport_transform(float x, float y, float width, float height) {
  // Calculate how to translate the x and y coordinates:
  float offset_x = (2.0 * x + (width - window_width)) / window_width;
  float offset_y = (2.0 * y + (height - window_height)) / window_height;

  // Calculate how to rescale the x and y coordinates:
  float scale_x = width / window_width;
  float scale_y = height / window_height;

  return glm::scale(glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));
}

'''

def test_viewport_transform( x,y,width,height,window_width,window_height):
  # Calculate how to translate the x and y coordinates:
    #offset_x = (2.0 * x + (width - window_width)) / window_width;
    #offset_y = (2.0 * y + (height - window_height)) / window_height;
    offset_x = (2.0 * x + (window_width  - width)) / window_width;
    offset_y = (2.0 * y + (window_height - height)) / window_height;

 # Calculate how to rescale the x and y coordinates:
    scale_x = width / window_width;
    scale_y = height / window_height;




#  return scale( translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));
    return np.array( [scale( translate( np.identity(4),offset_x, y=offset_y, z=0.0), scale_x,y=scale_y,z=1.0)  ],dtype=np.float32).reshape(4,4)
  

def viewport_transform( x,y,width,height,window_width,window_height):
  # Calculate how to translate the x and y coordinates:
    offset_x = (2.0 * x + (width - window_width)) / window_width;
    offset_y = (2.0 * y + (height - window_height)) / window_height;
   #offset_x = -1 + ( x + width ) / window_width;
   #offset_y = -1 + ( y + height) / window_height;
   # offset_x = -1.0 +  x  / window_width
   # offset_y = -1.0 +  y / window_height
    
    
    #offset_x = x / window_width;  #(2.0 * x + (window_width  - width)) / window_width;
    #offset_y = y /window_height; #(2.0 * y + (window_height - height)) / window_height;

 # Calculate how to rescale the x and y coordinates:
    scale_x = width / window_width;
    scale_y = height / window_height;

    #return scale( translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));


    #tmat = translate( np.identity(4),offset_x, y=offset_y, z=0.0)
    #t#mat = translate( np.identity(4),x, y=y, z=0.0)
    
   # print"tmat 1:"
   # print tmat    
    
   # print"tmat2:"
      
    tmat= np.array( [[ 1, 0, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, 0], [ offset_x, offset_y, 0.0, 1.0]],dtype=np.float32)
    #tmat= np.array( [[ 1, 0, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, 0], [ x, y, 0.0, 1.0]],dtype=np.float32)
   # print tmat    
    
    smat= scale(tmat, scale_x,y=scale_y,z=0.0)
   # print "scale mat"
   # print smat
    
   # return np.array( [scale( translate( np.identity(4),offset_x, y=offset_y, z=0.0), scale_x,y=scale_y,z=1.0)  ],dtype=np.float32).reshape(4,4)
   # return np.array( [scale( translate( np.identity(4),offset_x, y=offset_y, z=0.0), scale_x,y=scale_y,z=1.0)  ],dtype=np.float32).reshape(4,4)
   # return np.array( [scale( translate( np.identity(4),offset_x, y=offset_y, z=0.0), scale_x,y=scale_y,z=1.0)  ],dtype=np.float32).reshape(4,4)
    return smat



def translate(M, x, y=None, z=None):
    """
    translate produces a translation by (x, y, z) . 
    
    Parameters
    ----------
    x, y, z
        Specify the x, y, and z coordinates of a translation vector.
    """
    if y is None: y = x
    if z is None: z = x
  # T = np.array( [[ 1, 0, 0, x],[ 0, 1, 0, y],[ 0, 0, 1, z], [ 0, 0, 0, 1]],
  #                dtype=np.float32).T
    T = np.array( [[ 1, 0, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, 0], [ x, y, z, 1]],
                   dtype=np.float32)
                    
    return np.dot(M,T)    
    

def scale(M, x, y=None, z=None):
    """
    scale produces a non uniform scaling along the x, y, and z axes. The three
    parameters indicate the desired scale factor along each of the three axes.

    Parameters
    ----------
    x, y, z
        Specify scale factors along the x, y, and z axes, respectively.
    """
    if y is None: y = x
    if z is None: z = x
    S = [[ x, 0, 0, 0],
         [ 0, y, 0, 0],
         [ 0, 0, z, 0],
         [ 0, 0, 0, 1]]
    S = np.array(S,dtype=np.float32).T
    return  np.dot(M,S)


def xrotate(M,theta):
    t = math.pi*theta/180
    cosT = math.cos( t )
    sinT = math.sin( t )
    R = numpy.array(
        [[ 1.0,  0.0,  0.0, 0.0 ],
         [ 0.0, cosT,-sinT, 0.0 ],
         [ 0.0, sinT, cosT, 0.0 ],
         [ 0.0,  0.0,  0.0, 1.0 ]], dtype=np.float32)
    return np.dot(M,R)

def yrotate(M,theta):
    t = math.pi*theta/180
    cosT = math.cos( t )
    sinT = math.sin( t )
    R = numpy.array(
        [[ cosT,  0.0, sinT, 0.0 ],
         [ 0.0,   1.0,  0.0, 0.0 ],
         [-sinT,  0.0, cosT, 0.0 ],
         [ 0.0,  0.0,  0.0, 1.0 ]], dtype=np.float32)
    return np.dot(M,R)

def zrotate(M,theta):
    t = math.pi*theta/180
    cosT = math.cos( t )
    sinT = math.sin( t )
    R = numpy.array(
        [[ cosT,-sinT, 0.0, 0.0 ],
         [ sinT, cosT, 0.0, 0.0 ],
         [ 0.0,  0.0,  1.0, 0.0 ],
         [ 0.0,  0.0,  0.0, 1.0 ]], dtype=np.float32)
    return np.dot(M,R)


def rotate(M, angle, x, y, z, point=None):
    """
    rotate produces a rotation of angle degrees around the vector (x, y, z).
    
    Parameters
    ----------
    M
       Current transformation as a numpy array

    angle
       Specifies the angle of rotation, in degrees.

    x, y, z
        Specify the x, y, and z coordinates of a vector, respectively.
    """
    angle = math.pi*angle/180
    c,s = math.cos(angle), math.sin(angle)
    n = math.sqrt(x*x+y*y+z*z)
    x /= n
    y /= n
    z /= n
    cx,cy,cz = (1-c)*x, (1-c)*y, (1-c)*z
    R = numpy.array([[ cx*x + c  , cy*x - z*s, cz*x + y*s, 0],
                     [ cx*y + z*s, cy*y + c  , cz*y - x*s, 0],
                     [ cx*z - y*s, cy*z + x*s, cz*z + c,   0],
                     [          0,          0,        0,   1]]).T
    return np.dot(M,R)


def ortho( left, right, bottom, top, znear, zfar ):
    assert( right  != left )
    assert( bottom != top  )
    assert( znear  != zfar )
    
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = +2.0/(right-left)
    M[3,0] = -(right+left)/float(right-left)
    M[1,1] = +2.0/(top-bottom)
    M[3,1] = -(top+bottom)/float(top-bottom)
    M[2,2] = -2.0/(zfar-znear)
    M[3,2] = -(zfar+znear)/float(zfar-znear)
    M[3,3] = 1.0
    return M
        
def frustum( left, right, bottom, top, znear, zfar ):
    assert( right  != left )
    assert( bottom != top  )
    assert( znear  != zfar )

    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = +2.0*znear/(right-left)
    M[2,0] = (right+left)/(right-left)
    M[1,1] = +2.0*znear/(top-bottom)
    M[3,1] = (top+bottom)/(top-bottom)
    M[2,2] = -(zfar+znear)/(zfar-znear)
    M[3,2] = -2.0*znear*zfar/(zfar-znear)
    M[2,3] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    assert( znear != zfar )
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum( -w, w, -h, h, znear, zfar )
