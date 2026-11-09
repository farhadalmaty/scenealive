import json, os, shutil, subprocess, tempfile, unittest
import helpers, server_helpers as H
from scenelive.detect.blob import ColorBlobDetector
from scenelive.server.app import ServerApp, unpack_frame, WEB_DIR
from scenelive.server.config import ServerConfig
from scenelive.server.runner import ServerThread

APP_JS = os.path.join(WEB_DIR, "app.js"); NODE = shutil.which("node")


def node(script, *args, timeout=60):
    r = subprocess.run([NODE, "-e", script, APP_JS, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0: raise AssertionError("node failed:\n" + r.stderr[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node is not installed")
class TestPureHelpers(unittest.TestCase):
    def test_the_script_is_valid_javascript(self):
        r = subprocess.run([NODE, "--check", APP_JS], capture_output=True, text=True); self.assertEqual(r.returncode, 0, r.stderr)

    def test_compass_heading_from_device_orientation(self):
        out = node("const S=require(process.argv[1]); console.log(JSON.stringify([[0,90,0],[90,90,0],[180,90,0],[270,90,0],[45,90,0],[0,60,0],[0,120,0],[359,89,1]].map(a=>S.compassHeading(...a))))")
        exp = [0, 270, 180, 90, 315, 0, 0, 359]
        for got, want in zip(out, exp): self.assertAlmostEqual((got - want + 180) % 360 - 180, 0.0, delta=1.5, msg=(got, want))     # upright phone: alpha 0 = north, 90 = west

    def test_pack_frame_is_what_the_server_unpacks(self):
        hexs = node("const S=require(process.argv[1]); const b=S.packFrame({seq:3,t:1.5,yaw:null,w:640,h:360}, new Uint8Array([255,216,1,2,3]).buffer); console.log(JSON.stringify(Buffer.from(b).toString('hex')))")
        meta, jpeg = unpack_frame(bytes.fromhex(hexs)); self.assertEqual(meta, {"seq": 3, "t": 1.5}); self.assertEqual(jpeg, bytes([255, 216, 1, 2, 3]))

    def test_horizontal_fov_follows_the_orientation(self):
        a, b, c = node("const S=require(process.argv[1]); console.log(JSON.stringify([S.horizontalFov(69,640,360),S.horizontalFov(69,360,640),S.horizontalFov(100,1280,720)]))")
        self.assertAlmostEqual(a, 69.0, places=6); self.assertAlmostEqual(b, 42.3, delta=0.2); self.assertAlmostEqual(c, 100.0, places=6)     # portrait: the long side is vertical

    def test_camera_choice_ios_android_and_unlabelled(self):
        out = node("""const S=require(process.argv[1]); const v=(l,id)=>({kind:'videoinput',label:l,deviceId:id});
          const ios=[v('Front Camera','f'),v('Back Camera','m'),v('Back Ultra Wide Camera','u'),v('Back Telephoto Camera','t'),v('Back Dual Wide Camera','d'),{kind:'audioinput',label:'Back mic',deviceId:'a'}];
          const and=[v('camera2 1, facing front','1'),v('camera2 0, facing back','0'),v('camera2 2, facing back','2')];
          const id=(r)=>r.device?r.device.deviceId:null;
          console.log(JSON.stringify({iu:S.pickCameraDevice(ios,'ultrawide'),im:id(S.pickCameraDevice(ios,'main')),it:id(S.pickCameraDevice(ios,'tele')),au:S.pickCameraDevice(and,'ultrawide'),am:id(S.pickCameraDevice(and,'main')),at:S.pickCameraDevice(and,'tele'),none:S.pickCameraDevice([],'main')}))""")
        self.assertEqual((out["iu"]["device"]["deviceId"], out["iu"]["certain"], out["im"], out["it"]), ("u", True, "m", "t"))        # never the virtual 'Dual Wide'
        self.assertEqual((out["au"]["device"], out["au"]["certain"], out["am"], out["at"]["certain"], out["none"]["device"]), (None, False, "0", False, None))

    def test_speech_text_is_made_for_a_synthesizer(self):
        out = node("""const S=require(process.argv[1]); console.log(JSON.stringify([
          S.speechText('человек: прямо по курсу (12 ч), около 5 м, предварительно'), S.speechText('Вижу: машина: слева (9 ч), 10–15 м.'), S.speechText('дальше 15 м'),
          S.speechText('около 1 м'), S.speechText('около 21 м'), S.speechText('знак «стоп» ↔ уверенность 80 %'), S.speechText('Под ногами: слева асфальт, впереди гравий.'), S.speechText(''), S.speechText(null),
          S.metresWord(1), S.metresWord(2), S.metresWord(5), S.metresWord(11), S.metresWord(21), S.metresWord(2.5)]))""")
        self.assertEqual(out[:9], ["человек: прямо по курсу, около 5 метров, предварительно", "Вижу: машина: слева, от 10 до 15 метров.", "дальше 15 метров", "около 1 метра", "около 21 метра",
                                   "знак стоп уверенность 80 процентов", "Под ногами: слева асфальт, впереди гравий.", "", ""]); self.assertEqual(out[9:], ["метра", "метров", "метров", "метров", "метра", "метра"])

    def test_ground_chips_and_dynamic_russian_names(self):
        out = node("""const S=require(process.argv[1]); const g={left:{ru:'асфальт',quality:'good',quality_ru:'хорошее',bad:false},center:{ru:'яма на дороге',quality:'bad',quality_ru:'плохое',bad:true}};
          const before=S.nameRu('knife'); S.setNames({knife:'нож',person:'человек!'}); console.log(JSON.stringify({chips:S.groundChips(g),none:S.groundChips(null),before:before,after:S.nameRu('knife'),fallback:S.nameRu('zebra'),dyn:S.nameRu('person')}))""")
        self.assertEqual([c["zone"] for c in out["chips"]], ["слева", "впереди"]); self.assertTrue(out["chips"][1]["bad"]); self.assertFalse(out["chips"][0]["bad"]); self.assertEqual(out["none"], [])
        self.assertEqual((out["before"], out["after"], out["fallback"], out["dyn"]), ("knife", "нож", "zebra", "человек!"))                       # names from the server win over the built-in ones

    def test_the_radar_groups_duplicates_and_keeps_memory_apart(self):
        out = node("""const S=require(process.argv[1]); const e=(n,az,st,r)=>({name:n,rel_az_deg:az,status:st,range_m:r});
          const g=S.groupForRadar([e('person',2,'confirmed',5),e('person',10,'provisional',3),e('person',12,'confirmed',null),e('person',2,'remembered',5),e('knife',-80,'confirmed',2),e('door',50,'remembered',null)], {knife:'danger'});
          const many=[]; for(let i=0;i<60;i++) many.push(e('tree'+i,i*5-150,'confirmed',i)); console.log(JSON.stringify({g:g, n:S.groupForRadar(many,{}).length, empty:S.groupForRadar(null,{})}))""")
        by = {(m["name"], m["live"]): m for m in out["g"]}
        self.assertEqual(by[("person", True)]["count"], 3); self.assertEqual(by[("person", True)]["status"], "confirmed"); self.assertEqual(by[("person", True)]["range_m"], 3)         # one marker, nearest range
        self.assertEqual(by[("person", False)]["count"], 1); self.assertTrue(by[("knife", True)]["danger"]); self.assertFalse(by[("door", False)]["live"])                          # memory kept apart
        self.assertEqual(out["n"], 24); self.assertEqual(out["empty"], []); self.assertTrue(out["g"][0]["live"]); self.assertEqual(out["g"][0]["name"], "knife")                       # live first, then the nearest

    def test_staleness_check(self):
        out = node("const S=require(process.argv[1]); console.log(JSON.stringify([S.isStale(0,99999),S.isStale(1000,3000),S.isStale(1000,5100),S.isStale(1000,5100,10000),S.isStale(1000,4999)]))")
        self.assertEqual(out, [False, False, True, False, False])                                          # never before the first result; 4 s by default

    def test_the_page_warns_aloud_when_the_connection_or_the_results_stop(self):
        js = open(APP_JS, encoding="utf-8").read()
        for phrase in ("Связь с сервером потеряна. Не полагайтесь на подсказки.", "Подсказки не обновляются. Не полагайтесь на них.", "Связь с сервером восстановлена.", "Подсказки снова обновляются."): self.assertIn(phrase, js)

    def test_texts(self):
        out = node("const S=require(process.argv[1]); console.log(JSON.stringify([S.rangeText('~5 m'),S.rangeText('10-15 m'),S.rangeText('15+ m'),S.rangeText('?'),S.clockText(3,false),S.clockText(12,false),S.nameRu('person'),S.nameRu('zebra')]))")
        self.assertEqual(out, ["около 5 м", "10–15 м", "дальше 15 м", "расстояние неизвестно", "справа (3 ч)", "прямо по курсу (12 ч)", "человек", "zebra"])


@unittest.skipUnless(NODE, "node is not installed")
class TestIndependentClient(unittest.TestCase):
    def test_node_websocket_client_talks_to_the_server(self):
        d = tempfile.mkdtemp(); files = []
        for k in range(10):
            jpg, _ = H.jpeg_at(k / 3); p = os.path.join(d, f"f{k}.jpg"); open(p, "wb").write(jpg); files.append(p)
        app = ServerApp(ServerConfig(detector="blobs", agent="offline", https=False), ColorBlobDetector(), log=lambda *a: None)
        script = """
          const S=require(process.argv[1]); const fs=require('fs'); const port=process.argv[2]; const files=process.argv.slice(3);
          const ws=new WebSocket('ws://127.0.0.1:'+port+'/ws'); ws.binaryType='arraybuffer'; const out={results:[],chat:null,ready:null,errors:[]}; let k=0;
          const send=()=>{ const b=fs.readFileSync(files[k]); const ab=b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
            ws.send(S.packFrame({seq:k,t:k/3,yaw:(350+4*k/3)%360,w:640,h:360}, ab)); };
          ws.onopen=()=>ws.send(JSON.stringify({type:'hello',camera:'main',has_heading:true}));
          ws.onmessage=(ev)=>{ const m=JSON.parse(ev.data);
            if(m.type==='ready'){out.ready=m; send();}
            else if(m.type==='result'){ out.results.push({seq:m.seq,heading:m.heading,names:m.entries.map(e=>e.name),compute:m.compute_ms}); k++;
              if(k<files.length) send(); else ws.send(JSON.stringify({type:'chat',id:'q1',text:'Что вокруг?'})); }
            else if(m.type==='chat_reply'){ out.chat=m; console.log(JSON.stringify(out)); process.exit(0); }
            else if(m.type==='error'){ out.errors.push(m.message); } };
          ws.onerror=()=>{ console.log(JSON.stringify({fatal:'ws error'})); process.exit(2); };
          setTimeout(()=>{ console.log(JSON.stringify({fatal:'timeout',out})); process.exit(3); }, 40000);
        """
        with ServerThread(app) as th: out = node(script, th.port, *files)
        self.assertNotIn("fatal", out, out); self.assertEqual(out["errors"], []); self.assertEqual(out["ready"]["compass"], "client"); self.assertEqual(len(out["results"]), 10)
        self.assertAlmostEqual(out["results"][-1]["heading"], 12.0, delta=0.5)                                                        # 9 frames at 3 fps * 4 deg/s
        self.assertTrue({"person", "car", "pole"} <= set(out["results"][-1]["names"])); self.assertIn("человек", out["chat"]["text"]); self.assertEqual(out["chat"]["agent"], "offline")


if __name__ == "__main__": unittest.main()
