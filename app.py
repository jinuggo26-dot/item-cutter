"""Content-based kill-notification segmentation API.
No screenshot-size location, fixed percent ROI, or assumed y position.
Finds notification text and the adjacent bright weapon/icon artwork globally.
The foreground matte is estimated locally around the detected notification.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

app = FastAPI(title="Content-based killlog detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)
ROOT = Path(__file__).resolve().parent
GATE = asyncio.Semaphore(2)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 7_000_000
OUT_W, OUT_H = 760, 190

# Small appearance examples of the words "킬 알림" from real notifications.
# Templates contain only the rendered letters in a normalized brightness mask.
# They do not encode any screenshot location or screenshot dimensions.
TEMPLATE_PNG_B64 = ('iVBORw0KGgoAAAANSUhEUgAAAEEAAAAdCAAAAADQ0F4VAAADC0lEQVQ4EaXBf0icdRzA8ffHpz3rtluPe+pCsXlzSMpBtSGt+mMtENugH2Tpiob9HEiHgiGTKFmrMQoHhrlxTFhIgiwXG2zIyg0rdnRwUDoOSpmxee3oQM52dO5hZ8+efDy/q+Dk/rjXSyYsCzuTzdoeazL2S5I8jMYhm7xEEOEOj0V+LbFJViMlQkGB3b2gedPkJxTUOBNDq5phFUIhDWVD6BVXWY1QwN3tsW88gZ9YlVBA/eOjieoILnHIQyjgHfPi5h+SrE5YJg6uurEDx/iXONTWzZbLCP831vDKVygCfHgQ13g9g6/H/azYeKEOEhemjC5NXwReOIOrsxccug+jCFDVVDHn+YC5+zndmC4lx38NBt9gLnqzmfIkS5rMpOckoSDcMDr6UATYu8mT0Pv56CCnG+N+cq5XpEvZ8hvJL95nvnmcnJTZ1wEpMxhCESiZrsb14BVOvpx4gGXaTf3XAM0jRPaHYeg1oEa39Ql6O+GPso4+FAHMzZ2vwtgu+LJl/l5y2vrpvnUEnvs5QesAcH43rh1hSJntR1EE2PNpFWT883CqabqWFSfesnSN0WfXLbDvBKy75mNf/OpcGkiZrQMoUrbYehiY/uw48OMTw3tRul+q8PYcstm2feRPYLYSK6NpFsEzs5XtR1GkJ7nzeWA0PDN/Y23PQ6EDi6xoe7sqEz9nlRsl+9OA/3NN9/gM4n3DDsEQikxNag0GYKUtw4ShcHTaAgJGuqXFZ3k0DdgR5r9KD23btYAi5+3LZx85jhIZiFvZLGXV2YRol39nicNjUZRNqfXe7ebGU1dQZK2z6PDoFu67x1xf+Qzh9pgNuuH1Re/asJAFvH/x5CVcFddRjrWhCC4HZbwe5elvWVE7jWvrBC+etaFmilCb2OQILofhjzfot9JeM2KhvNtLR8SX1G2iLNs6geCaquk6giK4HEIDhp22dU+EOz55j66w18oYty+xbOf39H1taJlAP8EQUHKbJYLLQRl8E+Wp71ix5m9cZgrl4RiwZpElQrGEYgnFEoolFOsf79ELMz/+4NcAAAAASUVORK5CYII=', 'iVBORw0KGgoAAAANSUhEUgAAAEEAAAAdCAAAAADQ0F4VAAACOUlEQVQ4EZ3BUYTbYADA8f8nhFBCKOUoJZROrJRSjnsax7inPe1tNuaexs6e7mH2MNtejnEvm7GHsfftaYxjT31ZKSWUEgslhFhcCSH7llzTrG2uvdv9fkJyLQ4NCv0/tyssCMk1eIOdNku8SbVJTkiu5gSmzgrf1VrMCclVpmPFMlhnO1adjJBcwR3t6lwiGhgtUkKynT3rssEgaFdBSLYahbts1J/VLIRki+mwW2WL0NVMIdls4DeabDf2hWST6Xv/7l6Ff7waOafBwkxINhh9Mg51MnaLjK/q5CYmBSGBSFFZNznVXjE36JCJI53cxKQgfnYg1FnXn/oPVYj7e2C3uDCLtNlwn9TIoiBeHCsEBqtC24ydXVLf9sFukYncJrmRRUE8a9/HM1SWuU6tSYgOJAr0e2TGTRbGTQriy+enFhrLzvQOKa/GXL9HZmICsUqiMOhQEOfPVavXYEk0bnPBUepA7E9bFVJOgwW3TkHI0Umtc4/cmbpLIZhaLPErGrmJSUFI3toc17mEl+ywJlTjikLk1AwWhCR5ENyv3uESvqolYRInJrlhmxIh4fsRB4eaN50lYddiyaCeqJESVg3mArdNiZDA6Zvz35QFSZUVvt+iREhSt2wJJAorHKXOisBrkXG9LgUhSZ0cScrGlR1WBIFJiZCkgkcfDEomSoMV4aRDiZBkhoOz+kGPNZHGqlCnRMholtSAxz/aHzVuQPyKNUMDZq+/vutxA+I8VjWFlP1Eednh//0F5+vQuTu9NaUAAAAASUVORK5CYII=')
TEMPLATE_PNG_B64 += ('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',)
TEMPLATES = tuple(
    cv2.imdecode(np.frombuffer(base64.b64decode(s), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    for s in TEMPLATE_PNG_B64
)
SCALES = (0.65, 0.70, 0.84, 0.95, 1.00, 1.08, 1.22, 1.40, 1.65, 1.90, 2.20)


def _bright_letters(im: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    return (np.clip((val - 118.0) * 3.0, 0.0, 255.0)
            * np.clip((140.0 - sat) / 85.0, 0.0, 1.0)).astype(np.uint8)


def _icon_evidence(image: np.ndarray, x: int, y: int, scale: float) -> float:
    # The word appears alongside a weapon icon or decoration: reject identical
    # small words appearing in the game menu without the accompanying artwork.
    H, W = image.shape[:2]
    x0 = max(0, int(x - 178 * scale))
    x1 = min(W, int(x - 15 * scale))
    y0 = max(0, int(y - 10 * scale))
    y1 = min(H, int(y + 34 * scale))
    if x0 >= x1 or y0 >= y1:
        return -1.0
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    white_fraction = np.mean((hsv[:, :, 2] > 160) & (hsv[:, :, 1] < 100))
    saturated_fraction = np.mean((hsv[:, :, 2] > 110) & (hsv[:, :, 1] > 125))
    return float(min(white_fraction, .35) * 1.4 + min(saturated_fraction, .20) * .8)


def detect_notification(image: np.ndarray) -> tuple[int, int, float, float]:
    """Find the rendered kill-notification WORDS, not screenshot position or image ratio.

    A false location (the sidebar/menu) is worse than a failure, so candidates
    must pass BOTH independent tests: strong letter similarity and left-side
    weapon/icon evidence. The right-side banner evidence further confirms the
    horizontal notification layout. All measurements scale with detected text.
    """
    mask = _bright_letters(image)
    candidates = []
    for template in TEMPLATES:
        for scale in SCALES:
            w = max(12, round(template.shape[1] * scale))
            h = max(8, round(template.shape[0] * scale))
            if w >= mask.shape[1] or h >= mask.shape[0]:
                continue
            resized = cv2.resize(
                template, (w, h),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            )
            scores = cv2.matchTemplate(mask, resized, cv2.TM_CCOEFF_NORMED)
            # Search ALL pixels. Width/height are never used to select an ROI.
            for _ in range(5):
                _, match, _, (x, y) = cv2.minMaxLoc(scores)
                if match < 0.70:
                    break
                icon = _icon_evidence(image, x, y, scale)
                if icon >= 0.225:
                    candidates.append((float(match), float(icon), int(x), int(y), float(scale)))
                scores[max(0, y-round(45*scale)):y+round(45*scale)+1,
                       max(0, x-round(75*scale)):x+round(75*scale)+1] = -1.0

    if not candidates:
        raise ValueError(
            '킬 알림을 확실히 찾지 못했습니다. 메뉴나 무기 목록을 킬로그로 잘라내지 않도록 추출을 중단했습니다.'
        )
    # Word matching is decisive; bright sidebars are never rewarded solely for color.
    match, icon, x, y, scale = max(candidates, key=lambda c: (c[0] + min(c[1], .50)*.26, c[0]))

    # Reject screenshots *of the editing app* that contain a tiny example of a
    # notification on a checkerboard. This is based on the actual periodic
    # checker pattern, not on the input image dimensions or aspect ratio.
    H, W = image.shape[:2]
    x0, x1 = max(0, int(x+75*scale)), min(W, int(x+215*scale))
    y0, y1 = max(0, int(y-70*scale)), min(H, int(y-42*scale))
    if x1-x0 >= 50 and y1-y0 >= 14:
        context = cv2.cvtColor(image[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if (context.std() > 8.0
                and context.mean(axis=1).std() < 1.8
                and float(np.abs(context[:, :-max(1,round(17*scale))]
                            - context[:, max(1,round(17*scale)):]).mean()) > 8.0):
            raise ValueError('편집기 미리보기 화면이 아닌 원본 게임 스크린샷을 업로드해 주세요.')
    return x, y, scale, float(match)


def _local_crop(image: np.ndarray, anchor: tuple[int, int, float, float]):
    x, y, sc, _ = anchor
    H, W = image.shape[:2]
    # Bounds are offsets from the DETECTED TEXT and its measured glyph size.
    # They are not based on the full photo's width, height, or aspect ratio.
    x0 = max(0, int(round(x - 260 * sc)))
    x1 = min(W, int(round(x + 335 * sc)))
    y0 = max(0, int(round(y - 65 * sc)))
    y1 = min(H, int(round(y + 74 * sc)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("킬로그가 화면 가장자리에서 잘렸습니다.")
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1), (x-x0,y-y0,sc)


def _background(image: np.ndarray, rect: tuple[int,int,int,int], sc: float) -> np.ndarray:
    H, W = image.shape[:2]
    x0,y0,x1,y1=rect
    h,w=y1-y0,x1-x0
    t0=max(0,round(y0-10*sc)); t1=max(t0+1,round(y0-3*sc))
    b0=min(H-2,round(y1+3*sc)); b1=min(H,round(y1+11*sc))
    top=image[t0:t1,x0:x1]
    bottom=image[b0:b1,x0:x1]
    if top.size==0:top=image[y0:y0+1,x0:x1]
    if bottom.size==0:bottom=image[y1-1:y1,x0:x1]
    t=cv2.resize(cv2.GaussianBlur(top,(0,0),3),(w,1),interpolation=cv2.INTER_AREA)[0]
    b=cv2.resize(cv2.GaussianBlur(bottom,(0,0),3),(w,1),interpolation=cv2.INTER_AREA)[0]
    yy=np.arange(h,dtype=np.float32)[:,None,None]/max(1,h-1)
    bg=t.astype(np.float32)[None,:,:]*(1-yy)+b.astype(np.float32)[None,:,:]*yy
    return cv2.GaussianBlur(bg,(9,3),0).astype(np.uint8)


def _alpha_matte(image: np.ndarray, roi: np.ndarray, bg: np.ndarray, ax: float, ay: float, scale: float) -> np.ndarray:
    h,w=roi.shape[:2]
    fore_lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB).astype(np.float32)
    back_lab=cv2.cvtColor(bg,cv2.COLOR_BGR2LAB).astype(np.float32)
    diff=np.sqrt(((fore_lab-back_lab)**2).sum(axis=2))
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    sat=hsv[:,:,1].astype(np.float32)
    val=hsv[:,:,2].astype(np.float32)
    # Pixels with strong color/brightness contrast against local surroundings.
    strength=np.maximum.reduce([
        np.clip((diff-12)*7,0,255),
        np.clip((sat-75)*1.8,0,130)*np.clip((diff-6)/24,0,1),
        np.clip((val-130)*2.1,0,160)*np.clip((diff-9)/28,0,1),
    ])
    hard=(strength>47).astype(np.uint8)*255
    close_k=cv2.getStructuringElement(cv2.MORPH_RECT,(max(7,int(17*scale)//2*2+1),3))
    joined=cv2.morphologyEx(hard,cv2.MORPH_CLOSE,close_k)
    n,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    # Prioritize a long horizontal component close to the DETECTED GLYPH,
    # not the vertical game weapon, UI, or arbitrary colored particles.
    plausible=[]
    for i in range(1,n):
        cx,cy,cw,ch,area=map(int,stats[i])
        if cw < 70*scale or ch < 8*scale or area < 100*scale*scale:
            continue
        ydist=abs((cy+ch/2)-(ay+15*scale))
        xdist=max(0,(ax-175*scale)-(cx+cw),cx-(ax+200*scale))
        score=cw*3.0+area*.2-ydist*6.0-xdist*5.0
        plausible.append((score,i,cx,cy,cw,ch))
    if not plausible:
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
    # For the RIGHT straight bar, limit background beyond the top/bottom edge.
    # Detect the banner's actual horizontal edges using vertical color gradients,
    # rather than a percentage of the input screenshot or image dimensions.
    center_y=ay+14*scale
    left_band=int(max(0, ax+58*scale))
    right_band=int(min(w, ax+170*scale))
    if right_band-left_band >= 12:
        ys_int=np.arange(h)
        y_min=max(1,int(center_y-43*scale))
        y_max=min(h-2,int(center_y+57*scale))
        if y_max>y_min+12:
            # Median across bar columns reduces interference from white lettering.
            sample=cv2.GaussianBlur(roi[:,left_band:right_band,:],(9,5),0).astype(np.float32)
            intensity=cv2.cvtColor(sample.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
            profile=np.percentile(np.abs(np.diff(intensity,axis=0)),75,axis=1)
            upper_range=(max(1,int(center_y-40*scale)),max(2,int(center_y+2*scale)))
            lower_range=(max(1,int(center_y+8*scale)),min(h-2,int(center_y+55*scale)))
            if upper_range[1]>upper_range[0] and lower_range[1]>lower_range[0]:
                yt=int(np.argmax(profile[upper_range[0]:upper_range[1]])+upper_range[0])
                yb=int(np.argmax(profile[lower_range[0]:lower_range[1]])+lower_range[0])
                if yb-yt >= 12*scale and yb-yt <= 85*scale:
                    r_x=np.arange(w)[None,:]
                    r_y=np.arange(h)[:,None]
                    tail_area=r_x>=ax+32*scale
                    distance=np.maximum(yt-3*scale-r_y,r_y-(yb+3*scale))
                    # Soft fade with 3px source-image feather; preserve ornament area on left.
                    taper=np.clip(1-distance/max(2,3*scale),0,1)
                    alpha=np.where(tail_area,alpha*taper,alpha)
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
    return {'ok':True,'engine':'killlog-content-v9-strict','detail':'Use POST /extract-killlog'}


@app.get('/health')
def health():
    return {'ok':True,'engine':'killlog-content-v9-strict',
            'result_size':'760x190','endpoint':'POST /extract-killlog',
            'reference_endpoint':'POST /extract-killlog-reference',
            'detection':'full-frame text plus icon; strict rejection of menus, no size/ratio ROI'}


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
