#version 330 core
// Pass IDs supplied by GpuPreviewRenderer. Radial passes start at SOFT;
// source-profile corrections run before the added effect groups are composited.
const int SOFT = 2;
const int SOURCE_SHRINK = 4;
const int SOURCE_GROW = 5;
uniform vec2 outputSize, origin;
uniform float sampling, scale, cutoff, coefficient, widening, rayLength;
uniform float rainbow, frequency, phase, softSize, softPower;
uniform float ringPeak, innerWidth, outerWidth, innerPower, outerPower;
uniform int kind, gray;
uniform float softFilter;
uniform sampler2D originalImage;
flat in vec2 center, integerOrigin, direction;
flat in vec3 light;
flat in float filterWeight, measuredWidth;
flat in vec3 sourceBackground, sourceGradientX, sourceGradientY;
flat in vec3 sourcePeak;
out vec4 fragColor;
// Same path to white as render.fit_to_white: keep hue and luminance, shed saturation.
vec3 fitToWhite(vec3 c) {
    float peak=max(c.r,max(c.g,c.b));
    if(peak<=1.0) return max(c,0.0);
    float y=dot(c,vec3(0.212639004,0.715168655,0.072192311));
    float t=peak>y ? clamp((1.0-y)/(peak-y),0.0,1.0) : 0.0;
    return clamp(vec3(y)+t*(c-vec3(y)),0.0,1.0);
}
float point(float x,float a) { return pow(max(1.0-abs(x)*a,0.0),3.0); }
float fifteen(float x,float a) {
    if (x>1.5) return point(x,a);
    float sum=0.0;
    for(int j=0;j<15;j++) sum+=point(x+float(j)/14.0-0.5,a);
    return sum/15.0;
}
float primitive(float x,float a) {
    return sign(x)*(1.0-pow(max(1.0-a*abs(x),0.0),4.0))/(4.0*a);
}
float transverse(float cross,float a,float footprint) {
    if(sampling>1.0) return (primitive(cross+footprint/2.0,a)-primitive(cross-footprint/2.0,a))/footprint;
    if(cross>=1.5) return fifteen(cross,a);
    float sum=0.0;
    for(int j=-3;j<=3;j++) sum+=fifteen(cross+float(j)/8.0,a);
    return sum/7.0;
}
vec3 rainbowTint(vec3 c,float r) {
    if(rainbow<=0.0) return c;
    float u=r*scale*125000.0/pow(107.0-frequency,2.0);
    float theta=radians((floor(u)+36.0*(50.0-phase))*0.1);
    vec3 modulation=gray!=0 ? vec3(sin(theta)) : vec3(sin(theta),cos(theta),-sin(theta));
    return c*max(vec3(1.0)+0.017*rainbow*min(u/3000.0,1.0)*modulation,0.0);
}
// Ray light at one point, integrated across a square footprint of the given size.
vec3 rayLight(vec2 point,float footprint) {
    vec2 delta=point-center;
    float r=length(delta);
    bool horizontal=abs(direction.x)>=abs(direction.y);
    float along=horizontal ? point.x-integerOrigin.x : point.y-integerOrigin.y;
    bool positive=horizontal ? direction.x>0.0 : direction.y>0.0;
    float coverage=clamp((positive ? along : -along)/footprint+0.5,0.0,1.0);
    float dominant=horizontal ? abs(direction.x) : abs(direction.y);
    float slope=horizontal ? direction.y/direction.x : direction.x/direction.y;
    float cross=horizontal ? abs(delta.y-delta.x*slope) : abs(delta.x-delta.y*slope);
    float a=coefficient*dominant/(1.0+r*widening);
    float width=footprint*(1.0+abs(slope));
    float value=coverage*(primitive(cross+width/2.0,a)-primitive(cross-width/2.0,a))/width;
    value*=max(29.0*(pow(1.0+r*scale,-190.0/(rayLength+25.0))-0.000015),0.0);
    return rainbowTint(light*value,r);
}
float radialFlare(float r) {
    if(kind==SOFT) return max(5.7*pow(max(1.0-r/softSize,0.0),softPower)-0.00025,0.0);
    float inner=ringPeak*(1.0-(0.03+0.0097*innerWidth));
    float outer=ringPeak*(1.03+0.02*outerWidth);
    return r<ringPeak ? pow(clamp((r-inner)/(ringPeak-inner),0.0,1.0),innerPower)
        : pow(clamp((outer-r)/(outer-ringPeak),0.0,1.0),outerPower);
}
void main() {
    // OpenGL y runs upward; original-image y runs downward.
    vec2 world=origin+vec2(gl_FragCoord.x-0.5,outputSize.y-gl_FragCoord.y-0.5)*sampling;
    vec2 delta=world-center;
    float radius=length(delta);
    // These passes output positive correction magnitudes, not final RGB.
    // The renderer combines them with the source before Screen composition.
    if(kind>=SOURCE_SHRINK) {
        float sigma=measuredWidth/2.354820045;
        float u=dot(delta,delta)/(sigma*sigma);
        if(measuredWidth<=0.0 || u>=(kind==SOURCE_SHRINK ? 9.0 : 36.0)) discard;
        if(kind==SOURCE_GROW) {
            float wings=(exp(-u/4.5)-exp(-u/2.0))*pow(1.0-u/36.0,2.0);
            fragColor=vec4(softFilter*(2.0*filterWeight-1.0)*sourcePeak*wings,1.0);
            return;
        }
        vec3 original=texelFetch(originalImage,ivec2(gl_FragCoord.x,outputSize.y-gl_FragCoord.y),0).rgb;
        vec3 bg=sourceBackground+sourceGradientX*delta.x+sourceGradientY*delta.y;
        float fraction=.55*softFilter*(1.0-2.0*filterWeight)*(1.0-exp(-u/2.0))*pow(1.0-u/9.0,2.0);
        fragColor=vec4(fraction*max(original-max(bg,0.0),0.0),1.0);
        return;
    }
    if(radius>cutoff+(sampling>1.0 ? sampling*0.71 : 0.0)) discard;
    if(kind>=SOFT && sampling>1.0) {
        // Integrate emitted, clamped light, not a single point that can miss
        // the whole star. Full-resolution export/profile evaluation is unchanged.
        vec3 sum=vec3(0);
        for(int y=0;y<8;y++) for(int x=0;x<8;x++) {
            vec2 offset=(vec2(x,y)+0.5)/8.0-0.5;
            sum+=fitToWhite(light*radialFlare(length(delta+offset*sampling)));
        }
        fragColor=vec4(sum/64.0,1.0);
        return;
    }
    if(kind<SOFT && sampling>1.0) {
        // Zoomed out: average 4x4 sub-samples, each with its own steep falloff and
        // highlight limit, like a block of export pixels, instead of one point.
        vec3 sum=vec3(0);
        for(int y=0;y<4;y++) for(int x=0;x<4;x++) {
            vec2 offset=((vec2(x,y)+0.5)/4.0-0.5)*sampling;
            sum+=fitToWhite(rayLight(world+offset,sampling/4.0));
        }
        fragColor=vec4(sum/16.0,1.0);
        return;
    }
    float value;
    if(kind<SOFT) {
        bool horizontal=abs(direction.x)>=abs(direction.y);
        float along=horizontal ? world.x-integerOrigin.x : world.y-integerOrigin.y;
        bool positive=horizontal ? direction.x>0.0 : direction.y>0.0;
        float coverage=1.0;
        if(sampling==1.0) {
            if(positive ? along<0.0 : along>=0.0) discard;
        } else {
            coverage=clamp((positive ? along+sampling*0.5 : -along+sampling*0.5)/sampling,0.0,1.0);
        }
        float dominant=horizontal ? abs(direction.x) : abs(direction.y);
        float slope=horizontal ? direction.y/direction.x : direction.x/direction.y;
        float cross=horizontal ? abs(delta.y-delta.x*slope) : abs(delta.x-delta.y*slope);
        float a=coefficient*dominant/(1.0+radius*widening);
        value=coverage*transverse(cross,a,sampling*(1.0+abs(slope)));
        value*=max(29.0*(pow(1.0+radius*scale,-190.0/(rayLength+25.0))-0.000015),0.0);
    } else {
        value=radialFlare(radius);
    }
    // Per-source light is limited before the renderer blends overlapping quads.
    vec3 result=kind<SOFT ? rainbowTint(light*value,radius) : light*value;
    fragColor=vec4(fitToWhite(result),1.0);
}
