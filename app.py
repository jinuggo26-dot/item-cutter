        # Banner can be very pale; its alpha is initially conservative.
        chosen=np.where(strength>39,255,0).astype(np.uint8)
    else:
        plausible.sort(reverse=True)
        _,i,cx,cy,cw,ch=plausible[0]
        chosen=np.where(labels==i,255,0).astype(np.uint8)
        kern=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,11))
        expanded=cv2.dilate(chosen,kern)
        chosen=np.where((expanded>0)&(strength>16),255,chosen).astype(np.uint8)

    # Mask only the notification's vertical band and any adjoining decorations.
    # The source game weapon is typically LOWER; its disconnected large regions
    # must never be merged just because their colors resemble the notification.
    cy_ref=ay+14*scale
    ys=np.arange(h)[:,None]
    upper=cy_ref-58*scale
    lower=cy_ref+56*scale
    vertical=(ys>=upper)&(ys<=lower)
    # The right tail has a straight upper/lower edge, except ornaments near icon.
    # Trust contrast with background in the tail; do not fill opaque rectangles.
    xgrid=np.arange(w)[None,:]
    tail=xgrid>=ax+43*scale
    alpha=np.maximum(chosen.astype(np.float32), np.clip((strength-9)*3.1,0,255))
    alpha=np.where(vertical,alpha,0)
    # Strong negative priors for low-saturation screen background above/below bar.
    close_y=(ys>=cy_ref-23*scale)&(ys<=cy_ref+22*scale)
    alpha=np.where(tail & ~close_y, alpha*np.clip((strength-16)/38,0,1),alpha)
    # Remove isolated small background flecks; conserve colored particles adjacent to the bar.
    result=(alpha>24).astype(np.uint8)*255
    num,lab,st,_=cv2.connectedComponentsWithStats(result,8)
    for i in range(1,num):
        cx,cy,ww,hh,area=map(int,st[i])
        if area < max(14,int(22*scale*scale)):
            alpha[lab==i]=0
    return cv2.GaussianBlur(np.clip(alpha,0,255).astype(np.uint8),(3,3),0.6)


def extract_content(raw: bytes, reference: bool=False) -> bytes:
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPG, PNG 또는 WEBP 이미지를 읽을 수 없습니다.")
    H,W=image.shape[:2]
    if W*H>MAX_PIXELS or min(W,H)<160:
        raise ValueError("이미지가 지나치게 크거나 작습니다.")
    anchor=detect_notification(image)
    roi,rect,(ax,ay,scale)=_local_crop(image,anchor)
    bg=_background(image,rect,scale)
    alpha=_alpha_matte(image,roi,bg,ax,ay,scale)
    ys,xs=np.where(alpha>21)
    if len(xs)<45:
        raise ValueError("킬로그 이미지 부분을 구분하지 못했습니다.")
    # A single crop and a SINGLE transform shared between PNG and the original-color reference.
    x0=max(0,int(xs.min())-6); x1=min(roi.shape[1],int(xs.max())+7)
    y0=max(0,int(ys.min())-6); y1=min(roi.shape[0],int(ys.max())+7)
    color=roi[y0:y1,x0:x1]
    mask=alpha[y0:y1,x0:x1]
    if reference:
        mask=np.full_like(mask,255)
    ch,cw=color.shape[:2]
    s=min((OUT_W-16)/cw,(OUT_H-16)/ch)
    dw=max(1,round(cw*s));dh=max(1,round(ch*s))
    inter=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC
    color=cv2.resize(color,(dw,dh),interpolation=inter)
    mask=cv2.resize(mask,(dw,dh),interpolation=cv2.INTER_LINEAR)
    canvas=np.zeros((OUT_H,OUT_W,4),dtype=np.uint8)
    px=(OUT_W-dw)//2;py=(OUT_H-dh)//2
    canvas[py:py+dh,px:px+dw,:3]=color
    canvas[py:py+dh,px:px+dw,3]=mask
    ok,png=cv2.imencode('.png',canvas)
    if not ok:raise ValueError("PNG 파일을 만들지 못했습니다.")
    return png.tobytes()


@app.get('/')
def home():
    p=ROOT/'index.html'
    if p.is_file():return FileResponse(p,media_type='text/html')
    return {'ok':True,'engine':'killlog-content-v8','detail':'Use POST /extract-killlog'}


@app.get('/health')
def health():
    return {'ok':True,'engine':'killlog-content-v8',
            'result_size':'760x190','endpoint':'POST /extract-killlog',
            'reference_endpoint':'POST /extract-killlog-reference',
            'detection':'whole-image content matching; no screen-position percent ROI'}


async def _handle(file: UploadFile, reference: bool):
    raw=await file.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:
        raise HTTPException(413,detail='10MB 이하 이미지만 업로드할 수 있습니다.')
    async with GATE:
        try:
            png=await asyncio.to_thread(extract_content,raw,reference)
        except ValueError as exc:
            raise HTTPException(422,detail=str(exc))
    return Response(png,media_type='image/png',headers={'Cache-Control':'no-store'})


@app.post('/extract-killlog')
async def extract(file:UploadFile=File(...)):
    return await _handle(file,False)


@app.post('/extract-killlog-reference')
async def reference(file:UploadFile=File(...)):
    return await _handle(file,True)
