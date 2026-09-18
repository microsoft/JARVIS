(function(){
  const root=document.getElementById('hologram');
  const canvas=document.getElementById('holo-canvas');
  if(!root||!canvas)return;
  const context=canvas.getContext('2d');
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const points=Array.from({length:90},(_,index)=>({a:(index/90)*Math.PI*2,r:35+Math.random()*65,z:Math.random()*2-1,s:.3+Math.random()*1.6}));
  let width=0,height=0,phase=0;
  function resize(){const ratio=window.devicePixelRatio||1;width=canvas.clientWidth;height=canvas.clientHeight;canvas.width=width*ratio;canvas.height=height*ratio;context.setTransform(ratio,0,0,ratio,0,0);}
  function draw(){
    if(!width)resize(); context.clearRect(0,0,width,height); phase+=reduced?0:.008;
    const cx=width/2,cy=height*.48;
    points.forEach((point,index)=>{
      const angle=point.a+phase*(index%2?1:-.7), depth=Math.sin(angle)*point.z;
      const x=cx+Math.cos(angle)*point.r*(1+depth*.16), y=cy+Math.sin(angle)*point.r*.52+Math.sin(phase*2+index)*4;
      const alpha=.2+.5*(Math.sin(angle+phase)+1)/2;
      context.fillStyle=`rgba(99,231,255,${alpha})`;context.shadowBlur=8;context.shadowColor='#38dfff';
      context.beginPath();context.arc(x,y,point.s*(1+depth*.4),0,Math.PI*2);context.fill();
    });
    context.shadowBlur=0;if(!reduced)requestAnimationFrame(draw);
  }
  window.addEventListener('resize',resize);resize();draw();
  if(reduced) return;
  root.addEventListener('mousemove',event=>{
    const rect=root.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width-.5,y=(event.clientY-rect.top)/rect.height-.5;
    root.style.setProperty('--tilt-x',`${x*12}deg`);root.style.setProperty('--tilt-y',`${-y*8}deg`);
  });
})();
