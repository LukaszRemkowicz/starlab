#version 330 core
// Pass IDs supplied by GpuPreviewRenderer. Radial passes start at SOFT;
// source-profile corrections run before the added effect groups are composited.
const int SOFT = 2;
const int SOURCE_SHRINK = 4;
const int SOURCE_GROW = 5;
layout(location=0) in vec2 position;
layout(location=1) in vec3 color;
layout(location=2) in float brightness;
layout(location=3) in float sizeWeight;
layout(location=4) in float profileWidth;
layout(location=5) in vec3 background;
layout(location=6) in vec3 gradientX;
layout(location=7) in vec3 gradientY;
layout(location=8) in vec3 measuredPeak;
uniform vec2 outputSize, origin;
uniform float sampling, cutoff, coefficient, widening;
uniform float threshold, hue, saturation, intensity;
uniform float softFilter;
uniform int kind, rayCount, gray;
// Capacity for the fixed Secondary pattern; rayCount is the active count.
const int MAX_RAYS = 84;
uniform vec2 rayDirections[MAX_RAYS];
uniform float rayWeights[MAX_RAYS];
flat out vec2 center, integerOrigin, direction;
flat out vec3 light;
flat out float filterWeight, measuredWidth;
flat out vec3 sourceBackground, sourceGradientX, sourceGradientY;
flat out vec3 sourcePeak;
const vec2 corners[6] = vec2[6](vec2(0,-1),vec2(1,-1),vec2(1,1),vec2(0,-1),vec2(1,1),vec2(0,1));
void main() {
    int ray = gl_InstanceID % rayCount;
    direction = rayDirections[ray];
    // Positions are original-image pixel centers, independent of viewport zoom.
    integerOrigin = floor(position+0.5);
    center = floor(4.0*(position+0.5))/4.0-0.5;
    vec3 c = max(color + color.g*vec3(-hue/1500.0,hue/3000.0,-hue/1000.0),0.0);
    float sat = saturation>0.0 ? saturation/6.0 : saturation/150.0;
    c = max(c+sat*(c-(vec3(c.r+c.g+c.b)-c)/2.0),0.0);
    float luma=dot(color,vec3(0.212639004,0.715168655,0.072192311));
    c *= brightness/max(luma,1e-20);
    if (gray!=0) c=vec3((c.r+c.g+c.b)/3.0);
    light=c*max(brightness-threshold,0.0)/max((1.0-threshold)*brightness,1e-20)*intensity;
    light *= rayWeights[ray];
    filterWeight=sizeWeight; measuredWidth=profileWidth;
    sourceBackground=background; sourceGradientX=gradientX; sourceGradientY=gradientY;
    sourcePeak=measuredPeak;
    vec2 corner=corners[gl_VertexID];
    vec2 world;
    if (kind<SOFT) {
        bool horizontal=abs(direction.x)>=abs(direction.y);
        float dominant=horizontal ? abs(direction.x) : abs(direction.y);
        float slope=horizontal ? direction.y/direction.x : direction.x/direction.y;
        bool positive=horizontal ? direction.x>0.0 : direction.y>0.0;
        float t=mix(-sampling,cutoff+sampling,corner.x);
        float major=(horizontal ? integerOrigin.x : integerOrigin.y)+(positive ? t : -t);
        float middle=(horizontal ? center.y : center.x)+(major-(horizontal ? center.x : center.y))*slope;
        float halfWidth=(1.0+cutoff*widening)/(coefficient*dominant)+0.875+sampling;
        float minor=middle+corner.y*halfWidth;
        world=horizontal ? vec2(major,minor) : vec2(minor,major);
    } else {
        float extent=cutoff;
        if(kind>=SOURCE_SHRINK) { center=position; extent=(kind==SOURCE_SHRINK ? 3.0 : 6.0)*profileWidth/2.354820045; }
        world=center+vec2(corner.x*2.0-1.0,corner.y)*(extent+sampling);
    }
    // Half-pixel offset maps image centers to OpenGL fragment centers.
    vec2 pixel=(world-origin)/sampling+0.5;
    gl_Position=vec4(pixel.x/outputSize.x*2.0-1.0,1.0-pixel.y/outputSize.y*2.0,0,1);
}
