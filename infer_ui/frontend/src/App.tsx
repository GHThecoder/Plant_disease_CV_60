import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { predictFiles } from "./api";
import type { PredictItem, QueuedFile, TopHit } from "./types";

type ShownResult = PredictItem & { preview?: string };

const BAR_COLORS = ["#e2c48a", "#9cba8b", "#6f8f72", "#4d6654"];
const pct = new Intl.NumberFormat("tr-TR", {
  style: "percent",
  maximumFractionDigits: 1,
});

const KIND_TEXT: Record<TopHit["kind"], string> = {
  sağlıklı: "Sağlıklı",
  zararlı: "Zararlı",
  hastalık: "Hastalık",
};

const IMAGE_EXT = new Set([
  "jpg",
  "jpeg",
  "jfif",
  "png",
  "webp",
  "bmp",
  "gif",
  "tif",
  "tiff",
  "heic",
  "heif",
  "avif",
]);

const FILE_ACCEPT = [
  ".jpg",
  ".jpeg",
  ".jfif",
  ".png",
  ".webp",
  ".bmp",
  ".gif",
  ".tif",
  ".tiff",
  ".heic",
  ".heif",
  ".avif",
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "image/gif",
  "image/tiff",
].join(",");

function isImageFile(file: File) {
  if (file.type.toLowerCase().startsWith("image/")) return true;
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXT.has(ext);
}

export default function App() {
  const queueRef = useRef<HTMLElement>(null);
  const [offline, setOffline] = useState(false);
  const [queue, setQueue] = useState<QueuedFile[]>([]);
  const [results, setResults] = useState<ShownResult[] | null>(null);
  const urls = useRef<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      for (let attempt = 0; attempt < 8; attempt += 1) {
        try {
          const response = await fetch("/api/health");
          if (!response.ok) throw new Error("down");
          if (!cancelled) setOffline(false);
          return;
        } catch {
          await new Promise((resolve) => setTimeout(resolve, 400));
        }
      }
      if (!cancelled) setOffline(true);
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const created = urls.current;
    return () => {
      for (const url of created) URL.revokeObjectURL(url);
    };
  }, []);

  function addFiles(list: FileList | File[]) {
    const selected = Array.from(list);
    if (!selected.length) return;
    const images = selected.filter(isImageFile);
    if (!images.length) {
      setError("Fotoğraf seçilemedi. JPG, PNG veya WEBP deneyin.");
      return;
    }
    setError("");
    const added = images.map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2)}`,
      file,
      url: trackUrl(URL.createObjectURL(file)),
    }));
    setQueue((current) => [...current, ...added]);
  }

  useEffect(() => {
    if (queue.length === 0) return;
    queueRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [queue.length]);

  function trackUrl(url: string) {
    urls.current.push(url);
    return url;
  }

  function removeFile(id: string) {
    setQueue((current) => current.filter((item) => item.id !== id));
  }

  async function onPredict() {
    if (!queue.length || busy) return;
    setBusy(true);
    setError("");
    try {
      const payload = await predictFiles(queue.map((item) => item.file));
      setResults(
        payload.results.map((item, index) => ({
          ...item,
          preview: queue[index]?.url,
        })),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Tanı başarısız");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">Tarlada tanı</p>
        <h1>Fotoğraftan bitki durumu</h1>
        <p className="lede">
          Yaprak, başak veya meyve fotoğrafı yükleyin. İstediğiniz kadar kare
          ekleyebilirsiniz. Her fotoğraf için en olası dört sonuç, bitki adı ve
          durum adıyla yazılır.
        </p>
        <p className="crops">
          Elma, kiraz, mısır, pamuk, üzüm, şeftali, dolmalık biber, patates,
          pirinç, çilek, şeker kamışı, domates, buğday.
        </p>
      </header>

      {offline && (
        <p className="banner warn">
          Tanı servisi kapalı. Sayfayı bir süre sonra yenileyin.
        </p>
      )}

      <section
        className={dragOver ? "drop drop-active" : "drop"}
        onDragOver={(event) => {
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          addFiles(event.dataTransfer.files);
        }}
      >
        <div>
          <h2>Fotoğrafları buraya bırakın</h2>
          <p>Birden fazla fotoğraf seçebilirsiniz. Sayı sınırı yok.</p>
        </div>
        <div className="drop-actions">
          <label className="ghost file-btn">
            Dosya seç
            <input
              type="file"
              accept={FILE_ACCEPT}
              multiple
              onChange={(event) => {
                const picked = event.target.files ? Array.from(event.target.files) : [];
                event.target.value = "";
                addFiles(picked);
              }}
            />
          </label>
          <button type="button" className="solid" disabled={!queue.length || busy} onClick={onPredict}>
            {busy ? "Bakılıyor…" : `Sonucu gör (${queue.length})`}
          </button>
        </div>
      </section>

      {error && <p className="banner warn">{error}</p>}

      {queue.length > 0 && (
        <section className="queue" ref={queueRef}>
          <div className="section-head">
            <h2>Seçilen fotoğraflar</h2>
            <button
              type="button"
              className="text-btn"
              onClick={() => {
                setQueue([]);
                setResults(null);
              }}
            >
              Temizle
            </button>
          </div>
          <ul>
            {queue.map((item) => (
              <li key={item.id}>
                <img src={item.url} alt="" />
                <span>{item.file.name}</span>
                <button type="button" onClick={() => removeFile(item.id)} aria-label="Kaldır">
                  ×
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {results && results.length > 1 && <Summary results={results} />}

      {results && (
        <section className="cards">
          {results.map((item, index) => (
            <ResultCard key={`${item.filename}-${index}`} item={item} />
          ))}
        </section>
      )}
    </div>
  );
}

function Summary({ results }: { results: ShownResult[] }) {
  const stats = useMemo(() => {
    const ok = results.filter((item) => item.top4.length > 0);
    const healthy = ok.filter((item) => item.top4[0].healthy).length;
    const sick = ok.length - healthy;
    const crops = new Map<string, number>();
    for (const item of ok) {
      const crop = item.top4[0].crop;
      crops.set(crop, (crops.get(crop) ?? 0) + 1);
    }
    const cropRows = [...crops.entries()]
      .map(([crop, count]) => ({ crop, count }))
      .sort((a, b) => b.count - a.count);
    const topCrop = cropRows[0];
    return { ok: ok.length, healthy, sick, cropRows, topCrop };
  }, [results]);

  const sentence = stats.topCrop
    ? `${stats.ok} fotoğrafın ${stats.sick} tanesinde sorun görünüyor. En sık bitki: ${stats.topCrop.crop} (${stats.topCrop.count}).`
    : "Özet çıkarılacak sonuç yok.";

  return (
    <section className="summary">
      <div className="section-head">
        <h2>Hepsi bir arada</h2>
      </div>
      <p className="sentence">{sentence}</p>
      <div className="summary-grid">
        <div className="panel">
          <h3>Sağlıklı / sorunlu</h3>
          <div className="pie-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={[
                    { name: "Sağlıklı", value: stats.healthy },
                    { name: "Sorunlu", value: stats.sick },
                  ]}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={58}
                  outerRadius={84}
                  paddingAngle={3}
                  stroke="none"
                >
                  <Cell fill="#8eae78" />
                  <Cell fill="#c4846a" />
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [`${value} görsel`, name]}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="pie-legend">
              <span><i className="swatch healthy" /> Sağlıklı {stats.healthy}</span>
              <span><i className="swatch sick" /> Sorunlu {stats.sick}</span>
            </div>
          </div>
        </div>
        <div className="panel grow">
          <h3>Hangi bitki çıktı</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={stats.cropRows} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
              <CartesianGrid stroke="rgba(244,239,228,0.08)" vertical={false} />
              <XAxis dataKey="crop" tick={{ fill: "#d9d3c4", fontSize: 12 }} interval={0} />
              <YAxis allowDecimals={false} tick={{ fill: "#d9d3c4", fontSize: 12 }} width={28} />
              <Tooltip
                formatter={(value: number) => [`${value} görsel`, "Adet"]}
                cursor={{ fill: "rgba(244,239,228,0.05)" }}
              />
              <Bar dataKey="count" fill="#d4b483" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

function NameTick({
  x = 0,
  y = 0,
  payload,
}: {
  x?: number;
  y?: number;
  payload?: { value?: string };
}) {
  const text = payload?.value ?? "";
  const splitAt = text.indexOf(" - ");
  const crop = splitAt >= 0 ? text.slice(0, splitAt) : text;
  const condition = splitAt >= 0 ? text.slice(splitAt + 3) : "";
  return (
    <text x={x} y={y} textAnchor="end" fill="#efe8d8">
      <tspan x={x} dy={condition ? "-0.45em" : "0.32em"} fontSize={12}>
        {crop}
      </tspan>
      {condition ? (
        <tspan x={x} dy="1.2em" fontSize={11} fill="#c9c2b1">
          {condition}
        </tspan>
      ) : null}
    </text>
  );
}

function ResultCard({ item }: { item: ShownResult }) {
  const top = item.top4[0];
  const rows = item.top4.map((hit) => ({
    label: hit.label,
    probability: hit.probability,
  }));

  return (
    <article className={top ? (top.healthy ? "card healthy-card" : "card sick-card") : "card"}>
      {item.preview && <img className="preview" src={item.preview} alt={item.filename} />}
      <div className="card-body">
        <p className="filename">{item.filename}</p>
        {item.error || !top ? (
          <p className="banner warn">{item.error || "Tahmin yok"}</p>
        ) : (
          <>
            <div className="title-row">
              <div>
                <h3>{top.disease}</h3>
                <p>{top.crop}</p>
              </div>
              <div className="score">
                <strong>{pct.format(top.probability)}</strong>
                <span className={top.healthy ? "pill ok" : "pill bad"}>
                  {KIND_TEXT[top.kind]}
                </span>
              </div>
            </div>
            <h4>Olası durumlar</h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                layout="vertical"
                data={rows}
                margin={{ top: 8, right: 52, left: 8, bottom: 8 }}
              >
                <XAxis type="number" domain={[0, 1]} hide />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={196}
                  tick={<NameTick />}
                />
                <Tooltip
                  formatter={(value: number) => [pct.format(value), "Olasılık"]}
                />
                <Bar dataKey="probability" radius={5} barSize={22}>
                  {rows.map((row, index) => (
                    <Cell key={row.label} fill={BAR_COLORS[index] ?? BAR_COLORS[3]} />
                  ))}
                  <LabelList
                    dataKey="probability"
                    position="right"
                    formatter={(value: number) => pct.format(value)}
                    fill="#f4efe4"
                    fontSize={12}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
    </article>
  );
}
