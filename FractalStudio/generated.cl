//--- OpenCL specific header --------------------------------------
#define OPENCL
#ifdef FP64
//Double precision
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#define real double
#define complex double2
#else
//Single precision
#define real float
#define complex float2
#endif

#define rgba float4
#define in const
#define discard return (rgba)(0)

//Initialisers
//#define C(x,y) (complex)(x,y)
//Initialise complex,
//really strange problem when using (complex)(x,y) (eg: for power, passed to cpow() )
//setting components seems to work around it... (problem on NVIDIA Only)
complex C(in real x, in real y) { complex z; z.x = x; z.y = y; return z; }
#define R(x) (real)(x)

//Maths functions with alternate names
#define mod(a,b) fmod((real)a,(real)b)
#define abs(a) fabs(a)
#define inversesqrt(x) rsqrt(x)
#define PI  M_PI_F
#define E   M_E_F

//Palette lookup mu = [0,1]
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
#define gradient(mu) read_palette(palette, mu)
rgba read_palette(image2d_t palette, float mu)
{
  uint4 p = read_imageui(palette, sampler, (float2)(mu, 0.0));
  return (rgba)(p.x/255.0, p.y/255.0, p.z/255.0, p.w/255.0); 
}

#define CALCPIXEL rgba calcpixel(int iterations, complex coord, complex offset, bool julia, real pixelsize, complex dims, complex origin, complex selected, image2d_t palette, rgba background, __global real* input)

#define set_result(c) return clamp(c, 0.0f, 1.0f);
CALCPIXEL;  //Prototype

complex rotate2d(complex v, real angle)
{
  const real Cos = cos(radians(angle));
  const real Sin = sin(radians(angle));
  return (complex)(v.x * Cos - v.y * Sin, v.x * Sin + v.y * Cos);
}

//Converts a set of pixel coords relative to element into
// a new fractal pos based on current fractal origin, zoom & rotate...
complex convert(int2 pos, int2 size, real zoom, real rotation)
{
   real half_w = size.x * 0.5;
   real half_h = size.y * 0.5;

   //Scale based on smallest dimension and aspect ratio
   real box = size.x < size.y ? size.x : size.y;
   real scalex = size.x / box;
   real scaley = size.y / box;

   real re = scalex * (pos.x - half_w) / (half_w * zoom);
   real im = scaley * (pos.y - half_h) / (half_h * zoom);

   //Apply rotation to selected point
   return rotate2d((complex)(re, im), -rotation);
}

__kernel void sample(
    read_only image2d_t palette, 
    __global float4* temp,
    __global real* input, 
    int antialias,
    int julia,
    int iterations,
    int width,
    int height,
    int j, int k)
{
  real zoom = input[0];
  real rotation = input[1];
  real pixelsize = input[2];
  complex origin = (complex)(input[3],input[4]);
  complex selected = (complex)(input[5],input[6]);
  rgba background = (rgba)(input[7],input[8],input[9],input[10]);

  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  int2 size = (int2)(width, height);
  complex dims = (complex)(width, height);
  complex coord = origin + convert(pos, size, zoom, rotation);

  complex offset = (complex)((real)j/(real)antialias-0.5, (real)k/(real)antialias-0.5);
  rgba pixel = calcpixel(iterations, coord, offset, julia, pixelsize, 
                     dims, origin, selected, palette, background, input);

  if (j==0 && k==0) temp[get_global_id(1)*get_global_size(0)+get_global_id(0)] = (rgba)(0);
  temp[get_global_id(1)*get_global_size(0)+get_global_id(0)] += pixel;
}

__kernel void average(write_only image2d_t output, __global float4* temp, int passes)
{
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  rgba pixel = temp[get_global_id(1)*get_global_size(0)+get_global_id(0)];
  pixel /= (rgba)passes;
  write_imageui(output, (int2)(pos.x, pos.y), (uint4)(255*pixel.x,255*pixel.y,255*pixel.z,255*pixel.w));
}
//--- Maths function library --------------------------------------
#define zero(args) 0
#define czero(args) C(0.0,0.0)

real _inv(in real r)  {return 1.0/r;}
real _neg(in real x)  {return -x;}
real _sqr(in real x)  {return x*x;}
real _cube(in real x) {return x*x*x;}

bool equals(in complex z1, in complex z2, real tolerance)
{
  return distance(z1, z2) <= abs(tolerance);
}

real sgn(real x)
{
  return x / abs(x);
}

real manhattan(in complex z)
{
  return abs(z.x) + abs(z.y);
}

real norm(in complex z)
{
  //Norm squared
  return dot(z,z);
}

real cabs(in complex z)
{
  return length(z);
}

real arg(in complex z)
{
  return atan2(z.y,z.x);
}

real imag(in complex z)
{
  return z.y;
}

complex conj(in complex z)
{
  return C(z.x, -z.y);
}

complex mul(in complex a, in complex b)
{
  return C(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

complex div(in complex z, in complex w)
{
  return C(dot(z,w), z.y*w.x - z.x*w.y) / dot(w,w);
}

complex inv(in complex z)
{
  //1.0 / z
  return conj(z) / norm(z);
}

real lnr(in real r)
{
  //For colouring algorithms, return real part
  return log(abs(r));
}

real ln(in real x)
{
  return log(x);
}

complex cln(in complex z)
{
  return C(log(cabs(z)), arg(z));
}

complex clog10(in complex z)
{
  return C(log10(cabs(z)), arg(z));
}

complex neg(in complex z)
{
  return z * R(-1);
}

complex polar(in real r, in real theta)
{
  if (r < 0.0)
  {
    return C(r * cos(mod(theta+PI, R(2.0*PI))), -r * sin(theta));    
  }
  return C(r * cos(theta), r * sin(mod(theta, R(2.0*PI))));
}

complex sqr(in complex z)
{
  return C(z.x*z.x - z.y*z.y, z.x*z.y + z.y*z.x);
}

complex cube(in complex z)
{
  real x2 = z.x * z.x;
  real y2 = z.y * z.y;
  return C(z.x*x2 - z.x*y2 - z.x*y2 - y2*z.x, 
           x2*z.y + x2*z.y + z.y*x2 - y2*z.y);
}

complex cpow(in complex base, in complex exponent)
{
  //Simpler version? (Note: This is currently broken for non integer powers!)
  real r = cabs(base);
  real t = arg(base);
  real a = pow(r,exponent.x)*exp(-exponent.y*t);
  real b = exponent.x*t + exponent.y*log(r);
  return C(a*cos(b), a*sin(b));
}

complex cexp(in complex z) 
{
  real scalar = exp(z.x); // e^ix = cis 
  return C(scalar * cos(z.y), scalar * sin(z.y));
}

complex csqrt(in complex z)
{
  real ab = cabs(z);
  real x = (1.0 / sqrt(2.0)) * sqrt(ab + z.x);
  real y = (sgn(z.y) / sqrt(2.0)) * sqrt(ab - z.x);
  return C(x, y);
}

// Returns the sine of a complex number.
//    sin(z)  =  ( exp(i*z) - exp(-i*z) ) / (2*i)
complex csin(in complex z)
{
  //Using hyperbolic functions
  //sin(x + iy) = sin(x) cosh(y) + i cos(x) sinh(y)
  return C(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

// Returns the cosine of a complex number.
//     cos(z)  =  ( exp(i*z) + exp(-i*z) ) / 2
complex ccos(in complex z)
{
  //Using hyperbolic functions
  //cos(x + iy) = cos(x) cosh(y) - i sin(x) sinh(y)
  return C(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

// Returns the tangent of a complex number.
//     tan(z)  =  sin(z) / cos(z)
complex ctan(in complex z)
{
  return div(csin(z), ccos(z));
}

// Returns the principal arc sine of a complex number.
//     asin(z)  =  -i * log(i*z + sqrt(1 - z*z))
complex casin(in complex z)
{
  complex a = csqrt(C(1,0) - mul(z,z));
  a += C(-z.y, z.x); //z * i + a
  a = cln(a);
  return C(a.y, -a.x);  // a * -i
}

// Returns the principal arc cosine of a complex number.
//     acos(z)  =  -i * log( z + i * sqrt(1 - z*z) )
complex cacos(in complex z)
{
  complex a = csqrt(C(1,0) - mul(z,z));
  a = z + C(-a.y, a.x); //z + i * a
  a = cln(a);
  return C(a.y, -a.x);  // a * -i
}

// Returns the principal arc tangent of a complex number.
//     atan(z)  =  -i/2 * log( (i-z)/(i+z) )
complex catan(in complex z)
{
  complex a = div(C(0,1)-z, C(0,1)+z);
  return mul(C(0,-0.5), cln(a));  //-i/2 * log(a)
}

complex csinh(in complex z)
{
  //sinh(a+bi) = sinh(a) cos(b) + i(cosh(a) sin(b))
  return C(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y));
}

complex ccosh(in complex z)
{
  //cosh(a+bi) = cosh(a) cos(b) + i(sinh(a) sin(b))
  return C(cosh(z.x) * cos(z.y), sinh(z.x) * sin(z.y));
}

complex ctanh(in complex z)
{
  //tanh(z)  =  sinh(z) / cosh(z)
  return div(csinh(z), ccosh(z));
}

// Returns the principal inverse hyperbolic sine of a complex number.
//     asinh(z)  =  log(z + sqrt(z*z + 1))
complex casinh(in complex z)
{
  return cln(z + csqrt(mul(z,z) + C(1,0)));
}

// Returns the principal inverse hyperbolic cosine of a complex number.
//     acosh(z)  =  log(z + sqrt(z*z - 1))
complex cacosh(in complex z)
{
  return cln(z + csqrt(mul(z,z) - C(1,0)));
}

// Returns the principal inverse hyperbolic tangent of a complex number.
//     atanh(z)  =  1/2 * log( (1+z)/(1-z) )
complex catanh(in complex z)
{
  complex a = div(C(0,1)+z, C(0,1)-z);
  return mul(C(0.5,0), cln(a));
}

complex flip(in complex z)
{
  return C(z.y, z.x);
}


#define MAXITER 100

#define outside_set escaped || converged
//--- Main program ------------------------------------------------
#ifdef OPENCL
CALCPIXEL
{
#endif

//***DATA***
  //z(n+1) = 
  #define fractal_znext sqr(z) + c
  //Power (p)
  const real fractal_p = 2.0;
  //Escape
  const real fractal_escape = 4.0;
  //Bailout Test
  #define fractal_bailtest(args) norm(args)
  
  //Default colouring algorithm
  //Applies iteration count and repeat parameter to get a colour from the gradient palette

  //Palette repeat
  const real outside_colour_repeat = 1.0;
  


#ifdef GLSL
void main()
{
#endif

  //Globals
  complex z, c;
  complex point;            //Current point coord
  complex z_1;              //Value of z(n-1)
  complex z_2;              //Value of z(n-2)
  int count = 0;            //Step counter
  bool escaped = false;     //Bailout flags
  bool converged = false;
  bool perturb = false;     //Temporary: for old formulae

  int limit = iterations;   //Max iterations
  rgba colour = background;

  //Init fractal
  point = coord + C(offset.x*pixelsize, offset.y*pixelsize);

  //***INIT***


  if (julia)
  {
    //Julia set default
    z = point;
    c = selected;
  }
  else
  {
    //Mandelbrot set default
    z = C(0,0);
    c = point;
  }
  z_1 = z_2 = C(0,0);

  //Formula specific reset...
  //***RESET***


  //Iterate the fractal formula
  //(Loop counter can only be compared to constant in GL ES 2.0)
  for (int i=0; i < MAXITER; i++)
  {
    //Second iterations check: "limit" can be overridden to cut short iterations,
    //"iterations" must be a constant because of lame OpenGL ES 2.0 limitations on loops
    if (i == limit) break;
    if (i == iterations) break; //

    //Update z(n-2)
    z_2 = z_1;
    //Save current z value for z(n-1)
    z_1 = z;

    {
      //***PRE_TRANSFORM***

    }

    //Run next calc step
    count++;  //Current step count
    //***ZNEXT***

    z = fractal_znext;
  


    {
      //***POST_TRANSFORM***

    }

    //Check bailout conditions
    //***ESCAPED***

  escaped = (fractal_bailtest(z) > fractal_escape);
  

    //***CONVERGED***


    if (escaped || converged) break;

    //Colour calcs...
    {
      //***OUTSIDE_CALC***

    }
    {
      //***INSIDE_CALC***

    }
  }

  //Defined as: escaped || converged (or: true, when using same colour for both)
  if (outside_set)
  {
    //Outside colour: normalised colour index [0,1]
    //***OUTSIDE_COLOUR***

    colour = gradient(outside_colour_repeat * R(count) / R(limit));
  

  }
  else
  {
    //Inside colour: normalised colour index [0,1]
    //***INSIDE_COLOUR***

  }

  //***FILTER***


  //Set final colour
  set_result(colour);
}


