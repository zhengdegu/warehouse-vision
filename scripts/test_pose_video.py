# -*- coding: utf-8 -*-
"""Fight/Fall detection test - yolov8n-pose.pt + rule engine"""
import sys, os, argparse, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, torch
from ultralytics import YOLO
from src.vision.detector import Detection
from src.rules.anomaly import FightRule, FallRule

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def parse_pose(results):
    dets = []
    for r in results:
        bx, kps = r.boxes, r.keypoints
        if bx is None or len(bx)==0: continue
        for i in range(len(bx)):
            cid=int(bx.cls[i].item()); cf=float(bx.conf[i].item())
            x1,y1,x2,y2=bx.xyxy[i].tolist(); cx,cy=(x1+x2)/2,(y1+y2)/2
            tid=int(bx.id[i].item()) if bx.id is not None else -1
            kp=kps.data[i].cpu().numpy() if kps is not None and i<len(kps.data) else None
            dets.append(Detection(track_id=tid,class_id=cid,
                class_name="person" if cid==0 else str(cid),
                confidence=cf,bbox=[x1,y1,x2,y2],
                center=(cx,cy),foot=(cx,y2),keypoints=kp))
    return dets

def draw_skel(frame, dets):
    for d in dets:
        if d.keypoints is None: continue
        kp=d.keypoints
        for j in range(len(kp)):
            if kp[j][2]>0.3: cv2.circle(frame,(int(kp[j][0]),int(kp[j][1])),3,(0,255,0),-1)
        for a,b in SKELETON:
            if a<len(kp) and b<len(kp) and kp[a][2]>0.3 and kp[b][2]>0.3:
                cv2.line(frame,(int(kp[a][0]),int(kp[a][1])),(int(kp[b][0]),int(kp[b][1])),(0,255,0),2)

def draw_evt(frame, events):
    for e in events:
        sub=e.get("sub_type",""); det=e.get("detail",""); bb=e.get("bbox")
        c=(0,0,255) if sub=="fight" else (0,165,255) if sub=="fall" else (0,255,255)
        cv2.putText(frame,f"!! {sub.upper()} !!",(20,80),cv2.FONT_HERSHEY_SIMPLEX,2.0,c,4)
        cv2.putText(frame,det[:80],(20,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        if bb:
            b=[int(v) for v in bb]; cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),c,3)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--model",default="yolov8n-pose.pt")
    ap.add_argument("--conf",type=float,default=0.3)
    ap.add_argument("--output",default="")
    ap.add_argument("--show",action="store_true")
    ap.add_argument("--max-frames",type=int,default=0)
    ap.add_argument("--skip",type=int,default=2)
    ap.add_argument("--resize",type=int,default=480)
    args=ap.parse_args()

    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model} | Device: {device} | Conf: {args.conf}")
    print(f"Skip: {args.skip} | Resize: {args.resize} | MaxFrames: {args.max_frames}")
    model=YOLO(args.model); model.to(device)

    cap=cv2.VideoCapture(args.source)
    if not cap.isOpened(): print(f"Cannot open: {args.source}"); sys.exit(1)
    fps=int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps}fps, {total} frames, {total/fps:.1f}s")

    out_path=args.output or f"data/eval_videos/result_{os.path.basename(args.source)}"
    tmp_out=out_path+".tmp.mp4"
    writer=cv2.VideoWriter(tmp_out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))

    fight_r=FightRule(proximity_radius=500.0,min_speed=30.0,min_persons=2,confirm_frames=2,cooldown=5.0)
    fall_r=FallRule(ratio_threshold=0.8,min_ratio_change=0.3,min_y_drop=10.0,confirm_frames=1,cooldown=5.0)

    fc=0; pc=0; ft=0; flt=0; t0=time.time(); last_ann=None
    print(f"Output: {out_path}")
    print("="*70)

    try:
        while cap.isOpened():
            ret,frame=cap.read()
            if not ret: break
            fc+=1; ftime=fc/fps
            if fc%args.skip!=0:
                writer.write(last_ann if last_ann is not None else frame); continue
            if args.max_frames>0 and pc>=args.max_frames:
                writer.write(last_ann if last_ann is not None else frame); continue
            pc+=1

            inf=frame; sc=1.0
            if args.resize>0 and min(h,w)>args.resize:
                sc=args.resize/min(h,w); inf=cv2.resize(frame,(int(w*sc),int(h*sc)))
            res=model.track(inf,conf=args.conf,persist=True,tracker="bytetrack.yaml",
                            device=device,verbose=False,imgsz=args.resize if args.resize>0 else 640)
            dets=parse_pose(res)
            if sc!=1.0:
                for d in dets:
                    d.bbox=[v/sc for v in d.bbox]
                    cx,cy=d.center; d.center=(cx/sc,cy/sc)
                    fx,fy=d.foot; d.foot=(fx/sc,fy/sc)
                    if d.keypoints is not None: d.keypoints[:,0]/=sc; d.keypoints[:,1]/=sc

            evts=[]
            fe=fight_r.update(dets,"test",frame_ts=ftime); evts.extend(fe); ft+=len(fe)
            fae=fall_r.update(dets,"test",frame_ts=ftime); evts.extend(fae); flt+=len(fae)

            # Print first 10 frames in detail, event frames, and progress every 20
            show_it = (pc<=10) or len(evts)>0
            if show_it:
                print(f"\nFrame {fc}/{total}  t={ftime:.1f}s  persons={len(dets)}")
                for d in dets:
                    kn=int((d.keypoints[:,2]>0.3).sum()) if d.keypoints is not None else 0
                    print(f"  tid={d.track_id:>2d}  {d.class_name:<6s}  conf={d.confidence:.2f}  bbox=[{int(d.bbox[0]):>4d},{int(d.bbox[1]):>4d},{int(d.bbox[2]):>4d},{int(d.bbox[3]):>4d}]  kpts={kn}/17")
                for e in evts:
                    print(f"  >>> [{e['sub_type'].upper()}] {e['detail']}")
            elif pc%20==0:
                el=time.time()-t0; spd=pc/el if el>0 else 0
                print(f"  ... Frame {fc}/{total}, processed {pc}, {spd:.1f} fps, fight={ft} fall={flt}")

            ann=frame.copy()
            for d in dets:
                b=[int(v) for v in d.bbox]
                cv2.rectangle(ann,(b[0],b[1]),(b[2],b[3]),(255,255,0),2)
                cv2.putText(ann,f"id:{d.track_id} {d.class_name} {d.confidence:.2f}",
                            (b[0],b[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            draw_skel(ann,dets); draw_evt(ann,evts)
            el=time.time()-t0; spd=pc/el if el>0 else 0
            cv2.putText(ann,f"F:{fc}/{total} Det:{len(dets)} Fight:{ft} Fall:{flt} {spd:.1f}fps",
                        (10,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            last_ann=ann; writer.write(ann)
            if args.show:
                cv2.imshow("Detection",ann)
                if cv2.waitKey(1)&0xFF==ord("q"): break
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release(); writer.release()
        if args.show: cv2.destroyAllWindows()

    elapsed=time.time()-t0
    print("\n"+"="*70)
    print(f"Done: {fc} frames (processed {pc}), {elapsed:.1f}s ({pc/elapsed:.1f} fps)")
    print(f"Fight: {ft}  Fall: {flt}")
    print("Converting to H.264...")
    rc=subprocess.call(["ffmpeg","-y","-i",tmp_out,"-c:v","libx264","-preset","fast",
        "-crf","23","-pix_fmt","yuv420p",out_path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if rc==0:
        os.remove(tmp_out)
        print(f"Output: {out_path}")
    else:
        os.rename(tmp_out,out_path)
        print(f"ffmpeg failed, raw: {out_path}")

if __name__=="__main__":
    main()
