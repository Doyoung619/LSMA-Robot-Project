"""Render an English-only research demo from real BPM analysis outputs.

Run with the local .venv Python. Use --prepare to rerun the analyzer first.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'result' / '1_BPM'
RUN = OUT / 'analysis_run'
FIGURES = OUT / 'figures'
WORK = ROOT / 'tmp' / 'bpm_video_v2'
W, H, FPS = 1920, 1080, 30
PAPER, INK, MUTED = (255,255,255), (34,42,50), (94,103,110)
RULE, BLUE = (215,220,224), (37,79,113)
PX, PY = 120, 283
FONTS = {}
SCENES = [
    dict(key='input', title='BPM detection', duration=8,
         subtitle='Billie Jean, analyzed with the existing audio pipeline.',
         caption='The full track is analyzed offline. The next plots use the same 12-second excerpt.',
         source='rita_core/audio.py  /  analyze_audio()', figure='waveform.png'),
    dict(key='stft', title='Short-time Fourier transform', duration=12,
         subtitle='Each column shows the frequency content of a short audio window.',
         caption='The playhead follows the original audio. Bright vertical changes mark sharp attacks.',
         source='rita_core/audio.py  /  features()  /  scipy.signal.stft()', figure='stft.png'),
    dict(key='onset', title='Onset strength', duration=12,
         subtitle='Positive spectral changes are combined across three frequency bands.',
         caption='The weighted sum below is the onset signal used by the tempo estimator.',
         source='rita_core/audio.py  /  features()', figure='onset.png'),
    dict(key='tempo', title='Tempo candidates', duration=10,
         subtitle='Repeated onset intervals propose tempos between 70 and 180 BPM.',
         caption='This score proposes candidates. The final tempo is chosen by fitting a beat grid.',
         source='rita_robot/audio.py  /  analyze()  /  lag_score()', figure='tempo.png'),
    dict(key='grid', title='Beat-grid fit', duration=12,
         subtitle='The grid is initialized on 48 seconds, then refined across the full track.',
         caption='Orange lines are fitted beat times. Listen to how the grid follows the music.',
         source='rita_robot/audio.py  /  analyze()  /  soft-L1 regression', figure='beat_grid.png'),
    dict(key='native', title='Original diagnostic output', duration=8,
         subtitle='This figure is saved directly by the existing analysis code.',
         caption='Top: raw onset autocorrelation. Bottom: estimated beats over the first 12 seconds.',
         source='rita_core/audio.py  /  diagnostic_plot()  ->  tempo_diagnostic.png'),
    dict(key='result', title='Estimated tempo', duration=8,
         subtitle='The fitted period gives the tempo and a sequence of beat timestamps.',
         caption='One global tempo estimate; beat zero is an estimated origin, not a detected downbeat.',
         source='Actual analysis output  /  analysis_run/manifest.json'),
]
elapsed = 0.0
for scene in SCENES:
    scene['start'] = elapsed
    elapsed += scene['duration']
    scene['end'] = elapsed
DURATION = elapsed


def font(size, kind='body'):
    key = (size, kind)
    if key not in FONTS:
        name = {'body':'segoeui.ttf', 'bold':'segoeuib.ttf', 'mono':'consola.ttf'}[kind]
        FONTS[key] = ImageFont.truetype(str(Path('C:/Windows/Fonts') / name), size)
    return FONTS[key]


def text(d, xy, value, size=26, fill=INK, kind='body', anchor=None):
    d.text(xy, str(value), font=font(size,kind), fill=fill, anchor=anchor)


class Edit:
    def __init__(self):
        self.manifest = json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
        self.bpm = float(self.manifest['bpm'])
        self.period = float(self.manifest['period_seconds'])
        self.excerpt_start = float(self.manifest['excerpt_start_seconds'])
        self.boxes = self.manifest['plot_bounds']
        self.bases = [self.base(i) for i in range(len(SCENES))]

    def chrome(self,index):
        scene = SCENES[index]
        im = Image.new('RGB',(W,H),PAPER)
        d = ImageDraw.Draw(im)
        text(d,(88,44),'RITA  /  BPM DETECTION',23,BLUE,'mono')
        text(d,(1832,45),'OFFLINE AUDIO ANALYSIS',22,MUTED,'mono','ra')
        d.line((88,95,1832,95),fill=RULE,width=1)
        text(d,(88,125),scene['title'],58,INK,'bold')
        text(d,(90,207),scene['subtitle'],29,MUTED)
        d.line((88,929,1832,929),fill=RULE,width=1)
        text(d,(90,946),scene['caption'],27,INK)
        text(d,(90,998),scene['source'],20,MUTED,'mono')
        text(d,(1832,999),f'{index+1:02d} / {len(SCENES):02d}',20,MUTED,'mono','ra')
        return im

    def base(self,index):
        scene = SCENES[index]
        im = self.chrome(index)
        d = ImageDraw.Draw(im)
        if scene['key']=='native':
            plot = Image.open(RUN/'tempo_diagnostic.png').convert('RGB')
            plot.thumbnail((1280,650),Image.Resampling.LANCZOS)
            im.paste(plot,(90,266))
            d = ImageDraw.Draw(im)
            text(d,(1435,325),'SAVED BY THE CODE',21,BLUE,'mono')
            labels = ['tempo_diagnostic.png','','Original figure output.','','0.5x / 1x / 2x lines','show octave alternatives.']
            for j,label in enumerate(labels):
                text(d,(1435,378+j*43),label,23,INK)
            text(d,(1435,775),'Audio: track opening',22,MUTED)
            return im
        if scene['key']=='result':
            self.result(im)
            return im
        plot = Image.open(FIGURES/scene['figure']).convert('RGB')
        if plot.size != (1680,620):
            raise ValueError(f'Expected a 1680x620 figure: {scene["figure"]}: {plot.size}')
        im.paste(plot,(PX,PY))
        d = ImageDraw.Draw(im)
        if scene['key']=='input':
            text(d,(196,304),'INPUT',20,BLUE,'mono')
            text(d,(196,343),'Billie Jean.mp3',36,INK,'bold')
            text(d,(196,397),'294.01 s  /  stereo 44.1 kHz  /  original playback speed',24,MUTED)
        elif scene['key']!='tempo':
            text(d,(1794,256),f'Same excerpt  /  track {self.excerpt_start:.2f}-{self.excerpt_start+12:.2f} s',20,MUTED,'mono','ra')
        return im

    def result(self,im):
        d = ImageDraw.Draw(im)
        text(d,(115,310),f'{self.bpm:.2f}',132,BLUE,'bold')
        text(d,(130,476),'beats per minute',34,INK)
        text(d,(130,550),f'60 / {self.period:.8f} s',29,MUTED,'mono')
        text(d,(130,612),'Music playback stays at 1x.',27,MUTED)
        d.line((800,315,800,847),fill=RULE,width=1)
        text(d,(865,320),'ANALYZER VALUES',22,BLUE,'mono')
        rows = [('Period',f'{self.period:.8f} s'),
                ('Grid origin',f'{self.manifest["first_beat_seconds"]:.6f} s'),
                ('Fitted beats',str(self.manifest['beat_count'])),
                ('Audio analyzed',f'{self.manifest["duration_seconds"]:.2f} s')]
        for j,(label,value) in enumerate(rows):
            y=380+j*68
            text(d,(865,y),label,28,MUTED)
            text(d,(1718,y),value,29,INK,'mono','ra')
        text(d,(865,704),'First beat timestamps (track seconds)',26,INK)
        phase=float(self.manifest['first_beat_seconds'])
        beats=[phase+k*self.period for k in range(6)]
        text(d,(865,755),'  '.join(f'{b:.3f}' for b in beats[:3]),31,BLUE,'mono')
        text(d,(865,803),'  '.join(f'{b:.3f}' for b in beats[3:]),31,BLUE,'mono')

    def frame(self,t,transitions=True):
        index=next((i for i,s in enumerate(SCENES) if s['start']<=t<s['end']),len(SCENES)-1)
        scene=SCENES[index]
        local=t-scene['start']
        im=self.bases[index].copy()
        if scene['key'] in {'stft','onset','grid'}:
            axes=self.boxes[scene['key']]
            if isinstance(axes,dict): axes=[axes]
            overlay=Image.new('RGBA',im.size)
            od=ImageDraw.Draw(overlay)
            for box in axes:
                x=PX+box['x0']+(box['x1']-box['x0'])*min(local/12,1)
                y0,y1=PY+box['y0'],PY+box['y1']
                color=(255,255,255,235) if scene['key']=='stft' else (*INK,150)
                od.line((x,y0,x,y1),fill=color,width=2)
            im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
            text(ImageDraw.Draw(im),(1793,898),f'Track time  {self.excerpt_start+local:05.2f} s',18,MUTED,'mono','ra')
        d=ImageDraw.Draw(im)
        d.line((88,1050,1832,1050),fill=RULE,width=2)
        d.line((88,1050,88+1744*min(t/DURATION,1),1050),fill=BLUE,width=2)
        if transitions and index and local<.28:
            previous=self.frame(scene['start']-1/FPS,transitions=False)
            im=Image.blend(previous,im,max(0.,local/.28))
        return im

    def notes(self):
        chapters=[{**s,'audio_source_start_seconds':0.0 if s['key']=='native' else self.excerpt_start,
                   'audio_playback_rate':1.0} for s in SCENES]
        metadata=dict(version=2,language='en',duration_seconds=DURATION,resolution=[W,H],fps=FPS,
            audio_source='experiment/0_pre_and_etc/Billie Jean.mp3',
            audio_edit='Each chapter uses its corresponding source excerpt at 1x, with short fades.',
            narration=None,bpm=self.bpm,period_seconds=self.period,
            figure_provenance='analysis_run/manifest.json',original_code_figure='analysis_run/tempo_diagnostic.png',
            chapter_start_seconds=[s['start'] for s in SCENES],chapters=chapters,
            display_note='Offline full-track analysis. Cursors follow the audible 12-second excerpt; the original diagnostic shows the track opening.',
            limitations=['Single global tempo; 70-180 BPM candidate search.','Half/double-tempo ambiguity is possible.',
                         'Estimated grid origin is not a detected downbeat.','No ground-truth accuracy or runtime benchmark is presented.'])
        (OUT/'video_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        notes=(
            'RITA - BPM detection, revision 2\n\n'
            'BPM_Detection_Explainer.mp4\n70 seconds / 1920 x 1080 / 30 fps / English-only screen text\n\n'
            'Freshly computed figures from the local audio pipeline are used.\n'
            'The original diagnostic figure is included exactly as saved by diagnostic_plot().\n'
            'Original music plays at normal speed, with a brief fade at each edit. No synthetic narration.\n\n'
            f'Result: {self.bpm:.8f} BPM; {self.period:.9f} seconds per beat.\n'
            'STFT: mono 11025 Hz, window 1024, hop 128, logarithmic frequency axis.\n'
            'Onset bands: 35-180 / 180-2000 / 2000-5000 Hz; weights .55 / .30 / .15.\n'
            'Tempo candidates: weighted autocorrelation at 1x, 2x, 4x lags.\n'
            'Final selection: grid search, then robust full-track period/phase refinement.\n\n'
            'Caveats: one global tempo, octave ambiguity, no downbeat detection.\n'
            'This is not a ground-truth accuracy or processing-speed benchmark.\n\n'
            'Rebuild from experiment/0_pre_and_etc:\n'
            '  .venv\\Scripts\\python.exe prepare_bpm_assets.py --refresh\n'
            '  .venv\\Scripts\\python.exe render_bpm_video.py\n'
            'Preview only: add --preview to the renderer.\n\n'
            'Figures: figures/\nProvenance and actual output: analysis_run/\n'
            'Checks: verification.json\nPrevious video: previous_v1/\n')
        (OUT/'README.txt').write_text(notes,encoding='utf-8')
        def stamp(seconds):
            m=round(seconds*1000)
            return f'{m//3600000:02d}:{m//60000%60:02d}:{m//1000%60:02d},{m%1000:03d}'
        srt=[f'{i+1}\n{stamp(s["start"])} --> {stamp(s["end"])}\n{s["title"]}\n{s["subtitle"]}\n' for i,s in enumerate(SCENES)]
        (OUT/'BPM_Detection_Explainer.en.srt').write_text('\n'.join(srt),encoding='utf-8')

    def soundtrack(self):
        import soundfile as sf
        audio,sr=sf.read(ROOT/'Billie Jean.mp3',dtype='float32',always_2d=True)
        pieces=[]
        for scene in SCENES:
            start=0.0 if scene['key']=='native' else self.excerpt_start
            count=round(scene['duration']*sr)
            piece=audio[round(start*sr):round(start*sr)+count].copy()
            if len(piece)!=count: raise ValueError('Source audio is too short')
            n=round(.08*sr)
            piece[:n]*=np.linspace(0,1,n,dtype='float32')[:,None]
            piece[-n:]*=np.linspace(1,0,n,dtype='float32')[:,None]
            pieces.append(piece)
        target=WORK/'soundtrack.wav'
        sf.write(target,np.concatenate(pieces),sr,subtype='PCM_24')
        return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    if args.prepare: subprocess.run([sys.executable,str(ROOT/'prepare_bpm_assets.py'),'--refresh'],check=True)
    if not (RUN/'manifest.json').exists(): p.error('Run prepare_bpm_assets.py before rendering.')
    edit=Edit()
    edit.notes()
    sheet=Image.new('RGB',(1920,2160),(235,238,240))
    for i,scene in enumerate(SCENES):
        still=edit.frame(scene['start']+scene['duration']/2)
        still.save(WORK/f'scene_{i+1:02d}_{scene["key"]}.png')
        sheet.paste(still.resize((960,540),Image.Resampling.LANCZOS),((i%2)*960,(i//2)*540))
    sheet.save(OUT/'BPM_Storyboard.jpg',quality=94)
    edit.frame(47.5).save(OUT/'BPM_Preview.jpg',quality=95)
    if args.preview:
        print(f'Preview: {OUT / "BPM_Storyboard.jpg"}',flush=True)
        return
    soundtrack=edit.soundtrack()
    temporary=WORK/'BPM_Detection_Explainer.rendering.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-y',
         '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}',
         '-framerate',str(FPS),'-i','pipe:0','-i',str(soundtrack),
         '-map','0:v:0','-map','1:a:0','-t',str(DURATION),
         '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
         '-c:a','aac','-b:a','256k','-metadata','title=RITA - BPM detection',
         '-metadata:s:a:0','language=eng','-movflags','+faststart',str(temporary)]
    started=time.monotonic()
    with (WORK/'encode.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i in range(round(DURATION*FPS)):
                proc.stdin.write(edit.frame(i/FPS).tobytes())
                if i%(FPS*5)==0: print(f'Rendered {i/FPS:.0f}/{DURATION:.0f}s; elapsed {time.monotonic()-started:.1f}s',flush=True)
            proc.stdin.close()
            if proc.wait()!=0: raise RuntimeError((WORK/'encode.log').read_text(encoding='utf-8'))
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    shutil.move(str(temporary),str(OUT/'BPM_Detection_Explainer.mp4'))
    print(f'Saved {OUT / "BPM_Detection_Explainer.mp4"}',flush=True)


if __name__=='__main__':
    main()
