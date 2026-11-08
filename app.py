"""Relative-contrast kill-notification segmentation API, v9.1.
Deploy: replace the Render service app.py with this entire file, then redeploy.
Dependencies: fastapi, uvicorn, numpy, opencv-python-headless, python-multipart.
Start command: uvicorn app:app --host 0.0.0.0 --port $PORT.
Existing extract/reference/pair endpoints remain compatible.
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
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
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
TEMPLATES = tuple(
    cv2.imdecode(np.frombuffer(base64.b64decode(s), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    for s in TEMPLATE_PNG_B64
)
SCALES = (0.65, 0.70, 0.78, 0.84, 0.90, 0.93, 0.95, 1.00, 1.08, 1.16, 1.22, 1.40)


def _relative_mask(image: np.ndarray, weak: bool = False) -> np.ndarray:
    """Local brightness z-score, not an absolute 'white pixel' threshold."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), 7)
    variance = np.maximum(cv2.GaussianBlur((gray-mean)**2, (0, 0), 7), 0)
    contrast = (gray-mean)/np.maximum(np.sqrt(variance), 3.0)
    neutral = np.clip((210.0-hsv[:, :, 1])/140.0, .12, 1.0)
    noise_gate=np.clip((gray-mean-2.0)/8.0,0,1)
    return (np.clip((contrast-(.12 if weak else .35))*150, 0, 255)*neutral*noise_gate).astype(np.uint8)


def _glyph_evidence(mask: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
    patch=(mask[y:y+height,x:x+width]>45).astype(np.uint8)
    if not patch.size:return False
    _,_,stats,_=cv2.connectedComponentsWithStats(patch,8)
    chars=[r for r in stats[1:] if height*.18<=r[2]<=height*.8 and r[3]>=height*.24 and r[4]>=5]
    if len(chars)<3:return False
    span=max(r[0]+r[2] for r in chars)-min(r[0] for r in chars)
    return span>=width*.55


def _shape_layout(mask: np.ndarray, text_x: float, text_y: float, scale: float) -> float:
    h, w = mask.shape
    x0, x1 = max(0, round(text_x-190*scale)), min(w, round(text_x-12*scale))
    y0, y1 = max(0, round(text_y-12*scale)), min(h, round(text_y+40*scale))
    if x1-x0 < 12 or y1-y0 < 5:
        return 0.0
    region = mask[y0:y1, x0:x1]
    joined = cv2.morphologyEx((region > 45).astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    best = 0.0
    for x, y, cw, ch, area in stats[1:]:
        # A horizontally extended icon to the LEFT of the text, at a similar height.
        if cw < 23*scale or ch < 8*scale or area < 24*scale*scale or cw/max(ch, 1) < 1.6:
            continue
        alignment = max(0.0, 1-abs((y0+y+ch/2)-(text_y+14*scale))/max(28*scale, 1))
        fill = min(1., area/max(50*scale*scale, 1))
        best = max(best, float(alignment*fill*min(1., cw/max(60*scale, 1))))
    return best


def _edge_bands(image: np.ndarray, weak: bool = False) -> list[tuple[int, int, int, int, float]]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = np.linalg.norm(lab[1:]-lab[:-1], axis=2)
    threshold = max(3., float(np.mean(edge)+(0.7 if weak else 1.6)*np.std(edge)))
    binary = (edge > threshold).astype(np.uint8)*255
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = sorted((cv2.boundingRect(c) for c in contours), key=lambda r:r[2], reverse=True)
    lines = [r for r in lines if r[2] >= 80 and r[3] <= 8 and r[2]/max(r[3], 1) >= 12][:100]
    bands = []
    for x, y, width, height in lines:
        for bx, by, bw, bh in lines:
            gap = by-y
            overlap = min(x+width, bx+bw)-max(x, bx)
            if gap < 12 or gap > 145 or overlap < .7*min(width, bw):
                continue
            left, right = max(x, bx), min(x+width, bx+bw)
            if not 3 <= (right-left)/gap <= 24:
                continue
            score = min(1., overlap/max(width, bw))
            bands.append((left, y+1, right-left, gap-1, score))
    return sorted(bands, key=lambda b:b[4]*b[2], reverse=True)[:80]


def _layout_anchor(mask: np.ndarray, band: tuple):
    x, y, w, h, edge_score = band
    region = (mask[y:y+h, x:x+w] > 45).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    chars = [r for r in stats[1:] if 5 <= r[3] <= h*.85 and 2 <= r[2] <= r[3]*1.6 and r[4] >= 6]
    icons = [r for r in stats[1:] if r[2] >= 25 and r[3] >= 4 and r[2]/max(r[3], 1) >= 1.8 and r[4] >= 25]
    best = None
    for ix, iy, iw, ih, area in icons:
        right = [c for c in chars if c[0] >= ix+iw-2 and c[0] < ix+iw+max(120, 5*h)
                 and abs(c[1]+c[3]/2-(iy+ih/2)) < max(12, h*.45)]
        if len(right) < 3:
            continue
        right.sort(key=lambda r:r[0])
        glyph_h = float(np.median([r[3] for r in right]))
        scale = max(.35, glyph_h/24.)
        tx, ty = x+right[0][0], y+min(c[1] for c in right)
        layout = _shape_layout(mask, tx, ty, scale)
        if layout < .3:
            continue
        score = .25*edge_score+.35*layout+.15*min(len(right)/6, 1)
        if best is None or score > best[0]:
            best = (score, tx, ty, scale)
    return best


def detect_candidates(image: np.ndarray) -> list[dict]:
    """Full image search: paired horizontal edges + icon/text layout; at most 8 distinct regions."""
    H, W = image.shape[:2]
    resize = min(1., 1600/max(H, W))
    work = cv2.resize(image, (round(W*resize), round(H*resize)), interpolation=cv2.INTER_AREA) if resize < 1 else image
    proposals = []
    # Strict pass, followed by a weaker LOCAL-CONTRAST pass for dark banners.
    for weak in (False, True):
        mask = _relative_mask(work, weak)
        bands = _edge_bands(work, weak)
        for template in TEMPLATES:
            if template is None:
                continue
            for scale in SCALES:
                w, h = max(12, round(template.shape[1]*scale)), max(8, round(template.shape[0]*scale))
                if w >= mask.shape[1] or h >= mask.shape[0]:
                    continue
                scaled = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                response = cv2.matchTemplate(mask, scaled, cv2.TM_CCOEFF_NORMED)
                for _ in range(8):
                    _, match, _, (x, y) = cv2.minMaxLoc(response)
                    if match < (.36 if weak else .40):
                        break
                    layout = _shape_layout(mask, x, y, scale)
                    nearby = max((b[4] for b in bands if b[0]-30*scale <= x <= b[0]+b[2]
                                  and b[1]-25*scale <= y <= b[1]+b[3]), default=0.)
                    if layout >= .25 and (_glyph_evidence(mask,x,y,w,h) or match >= .72) and (nearby >= .4 or match >= .54):
                        score = .85*float(match)+.10*layout+.05*nearby
                        proposals.append((score, x/resize, y/resize, scale/resize, 'text+icon+edges' if nearby else 'text+icon'))
                    response[max(0,y-h):y+h+1, max(0,x-w):x+w+1] = -1
        # Geometry fallback does not require exactly the same rendered Korean font.
        for band in bands:
            result = _layout_anchor(mask, band)
            if result:
                score, x, y, scale = result
                proposals.append((score, x/resize, y/resize, scale/resize, 'edges+icon+text-shapes'))
    selected = []
    for score, x, y, scale, method in sorted(proposals, reverse=True):
        if score < .42:
            continue
        rect = (max(0, round(x-260*scale)), max(0, round(y-65*scale)),
                min(W, round(x+335*scale)), min(H, round(y+74*scale)))
        if rect[2]-rect[0] < 40 or rect[3]-rect[1] < 10:
            continue
        duplicate = False
        for candidate in selected:
            a, b = rect, candidate['box']
            intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if intersection/max(union,1) > .42 or (abs(x-candidate['anchor'][0]) < max(40,60*scale) and abs(y-candidate['anchor'][1]) < max(20,30*scale)):
                duplicate = True
                break
        if not duplicate:
            selected.append({'anchor':(round(x),round(y),scale,score), 'box':rect,
                             'score':round(float(score),4), 'method':method})
        if len(selected) == 8:
            break
    if not selected:
        raise ValueError('가로 경계와 총 그림·오른쪽 글자 배치를 찾지 못했습니다. 원본 게임 화면 또는 수동 선택을 이용해 주세요.')
    return selected


def detect_notification(image: np.ndarray) -> tuple[int,int,float,float]:
    return detect_candidates(image)[0]['anchor']


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
    """Matte the measured horizontal body, then retain only adjoining left ornaments."""
    h,w=roi.shape[:2]
    lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB).astype(np.float32)
    # Measure edges beside the letters, where glyph strokes cannot become the border.
    l=max(0,round(ax+72*scale));r=min(w,round(ax+200*scale))
    if r-l<12:l=max(0,round(ax));r=min(w,round(ax+100*scale))
    vertical=np.linalg.norm(lab[1:,l:r]-lab[:-1,l:r],axis=2).mean(axis=1)
    def edge(lo,hi,fallback):
        lo=max(0,min(h-2,round(lo)));hi=max(lo+1,min(h-1,round(hi)))
        values=vertical[lo:hi]
        if not values.size or float(values.max())<2:return max(0,min(h-1,round(fallback)))
        return lo+int(np.argmax(values))+1
    top=edge(ay-22*scale,ay+4*scale,ay-8*scale)
    bottom=edge(ay+25*scale,ay+48*scale,ay+35*scale)
    if bottom-top<20*scale:top=max(0,round(ay-8*scale));bottom=min(h-1,round(ay+35*scale))
    # Estimate the screen immediately above/below the actual body, not around a large ROI.
    thickness=max(2,round(4*scale))
    a=roi[max(0,top-thickness-2):max(1,top-1)]
    b=roi[min(h-1,bottom+1):min(h,bottom+thickness+2)]
    if not a.size:a=roi[:1]
    if not b.size:b=roi[-1:]
    upper=cv2.GaussianBlur(np.median(a,axis=0).astype(np.float32)[None,:,:],(0,0),5)[0]
    lower=cv2.GaussianBlur(np.median(b,axis=0).astype(np.float32)[None,:,:],(0,0),5)[0]
    mix=np.clip((np.arange(h,dtype=np.float32)-top)/max(bottom-top,1),0,1)[:,None,None]
    local_bg=(upper[None,:,:]*(1-mix)+lower[None,:,:]*mix).astype(np.uint8)
    local_lab=cv2.cvtColor(local_bg,cv2.COLOR_BGR2LAB).astype(np.float32)
    diff=np.linalg.norm(lab-local_lab,axis=2)
    outer_diff=np.linalg.norm(lab-cv2.cvtColor(bg,cv2.COLOR_BGR2LAB).astype(np.float32),axis=2)
    # A small relative difference can be meaningful on a dark banner.
    body_alpha=np.clip((diff-5)*18,0,255)
    ys=np.arange(h)[:,None];xs=np.arange(w)[None,:]
    body_band=(ys>=top)&(ys<bottom)
    body=np.where(body_band,body_alpha,0)
    # White icon/text and a continuous body seed the connected foreground.
    binary=(body>38).astype(np.uint8)
    joined=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(max(3,round(7*scale)),3)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    chosen=np.zeros((h,w),np.uint8)
    for i in range(1,n):
        x,y,cw,ch,area=map(int,stats[i])
        if x<=ax+70*scale and x+cw>=ax-155*scale and area>35*scale*scale:
            chosen[labels==i]=1
    # Expand only around that connected body; eliminate unrelated background patches.
    support=cv2.dilate(chosen,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,5)))
    alpha=np.where((support>0)&body_band,body,0)
    # Decorative emblems are left of the glyphs and can extend above/below the body.
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    saturation=hsv[:,:,1].astype(np.float32)
    decoration=(outer_diff>16)&((saturation>65)|(np.max(roi.astype(np.float32),axis=2)-np.max(bg.astype(np.float32),axis=2)>18))
    left=(xs<ax-18*scale)&(xs>ax-220*scale)
    decoration=(decoration&left).astype(np.uint8)
    connected=cv2.morphologyEx(decoration,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(connected,8)
    for i in range(1,n):
        x,y,cw,ch,area=map(int,stats[i])
        overlap=np.any((labels==i)&(cv2.dilate(chosen,np.ones((9,9),np.uint8))>0))
        if overlap and area>25*scale*scale:
            region=cv2.dilate((labels==i).astype(np.uint8),np.ones((3,3),np.uint8))>0
            alpha=np.where(region&left,np.maximum(alpha,np.clip((outer_diff-8)*18,0,255)),alpha)
    return cv2.GaussianBlur(np.clip(alpha,0,255).astype(np.uint8),(3,3),.5)


def _decode(raw: bytes) -> np.ndarray:
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('JPG, PNG 또는 WEBP 이미지를 읽을 수 없습니다.')
    H,W=image.shape[:2]
    if W*H>MAX_PIXELS or min(W,H)<160:
        raise ValueError('이미지가 지나치게 크거나 작습니다.')
    return image


def _pair_from_image(image: np.ndarray, selected: dict) -> tuple[np.ndarray,np.ndarray]:
    roi,rect,(ax,ay,scale)=_local_crop(image,selected['anchor'])
    bg=_background(image,rect,scale)
    alpha=_alpha_matte(image,roi,bg,ax,ay,scale)
    ys,xs=np.where(alpha>21)
    if len(xs)<45:
        raise ValueError('킬로그 이미지 부분을 구분하지 못했습니다. 다른 감지 후보를 선택해 주세요.')
    x0=max(0,int(xs.min())-6);x1=min(roi.shape[1],int(xs.max())+7)
    y0=max(0,int(ys.min())-6);y1=min(roi.shape[0],int(ys.max())+7)
    color,mask=roi[y0:y1,x0:x1],alpha[y0:y1,x0:x1]
    ch,cw=color.shape[:2];ratio=min((OUT_W-16)/cw,(OUT_H-16)/ch)
    dw,dh=max(1,round(cw*ratio)),max(1,round(ch*ratio))
    color=cv2.resize(color,(dw,dh),interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_CUBIC)
    mask=cv2.resize(mask,(dw,dh),interpolation=cv2.INTER_LINEAR)
    cut=np.zeros((OUT_H,OUT_W,4),dtype=np.uint8);ref=np.zeros_like(cut)
    px,py=(OUT_W-dw)//2,(OUT_H-dh)//2
    cut[py:py+dh,px:px+dw,:3]=color;cut[py:py+dh,px:px+dw,3]=mask
    ref[py:py+dh,px:px+dw,:3]=color;ref[py:py+dh,px:px+dw,3]=255
    return cut,ref


def _png(image: np.ndarray) -> bytes:
    ok,data=cv2.imencode('.png',image)
    if not ok:raise ValueError('PNG 파일을 만들지 못했습니다.')
    return data.tobytes()


def extract_pair(raw: bytes, candidate: int=0) -> tuple[np.ndarray,np.ndarray]:
    image=_decode(raw);candidates=detect_candidates(image)
    if not 0<=candidate<len(candidates):raise ValueError('존재하지 않는 감지 후보입니다. 후보를 다시 확인하세요.')
    return _pair_from_image(image,candidates[candidate])


def extract_content(raw: bytes, reference: bool=False, candidate: int=0) -> bytes:
    cut,ref=extract_pair(raw,candidate)
    return _png(ref if reference else cut)


def extract_bundle(raw: bytes, candidate: int=0) -> bytes:
    cut,ref=extract_pair(raw,candidate)
    return _png(np.concatenate((cut,ref),axis=0))


def candidate_bundle(raw: bytes) -> dict:
    image=_decode(raw);candidates=detect_candidates(image);out=[]
    for index,c in enumerate(candidates):
        x0,y0,x1,y1=c['box'];roi=image[y0:y1,x0:x1]
        h,w=roi.shape[:2];ratio=min(760/w,190/h)
        preview=cv2.resize(roi,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
        out.append({'index':index,'score':c['score'],'method':c['method'],'box':list(c['box']),
                    'preview':'data:image/png;base64,'+base64.b64encode(_png(preview)).decode('ascii')})
    return {'engine':'killlog-relative-v9.1','candidates':out,'limit':8}


@app.get('/')
def home():
    p=ROOT/'index.html'
    if p.is_file():return FileResponse(p,media_type='text/html')
    return {'ok':True,'engine':'killlog-relative-v9.1','detail':'Use POST /extract-killlog'}


@app.get('/health')
def health():
    return {'ok':True,'engine':'killlog-relative-v9.1','result_size':'760x190',
            'endpoint':'POST /extract-killlog','reference_endpoint':'POST /extract-killlog-reference',
            'pair_endpoint':'POST /extract-killlog-pair','candidate_endpoint':'POST /extract-killlog-candidates',
            'max_candidates':8,'detection':'global horizontal edge pairs + icon/text layout + local relative brightness'}


async def _read(file: UploadFile):
    raw=await file.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise HTTPException(413,detail='10MB 이하 이미지만 업로드할 수 있습니다.')
    return raw


async def _handle(file: UploadFile, kind: str, candidate: int=0):
    raw=await _read(file)
    async with GATE:
        try:
            if kind=='candidates':return await asyncio.to_thread(candidate_bundle,raw)
            if kind=='pair':png=await asyncio.to_thread(extract_bundle,raw,candidate)
            else:png=await asyncio.to_thread(extract_content,raw,kind=='reference',candidate)
        except ValueError as exc:raise HTTPException(422,detail=str(exc))
    return Response(png,media_type='image/png',headers={'Cache-Control':'no-store'})


@app.post('/extract-killlog')
async def extract(file:UploadFile=File(...),candidate:int=Query(0,ge=0,le=7)):
    return await _handle(file,'cut',candidate)


@app.post('/extract-killlog-reference')
async def reference(file:UploadFile=File(...),candidate:int=Query(0,ge=0,le=7)):
    return await _handle(file,'reference',candidate)


@app.post('/extract-killlog-pair')
async def pair(file:UploadFile=File(...),candidate:int=Query(0,ge=0,le=7)):
    return await _handle(file,'pair',candidate)


@app.post('/extract-killlog-candidates')
async def candidates(file:UploadFile=File(...)):
    return await _handle(file,'candidates')
