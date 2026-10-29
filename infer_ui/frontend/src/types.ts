export type TopHit = {
  rank: number;
  class_name: string;
  class_index: number;
  label: string;
  crop: string;
  disease: string;
  healthy: boolean;
  kind: "sağlıklı" | "zararlı" | "hastalık";
  probability: number;
};

export type PredictItem = {
  filename: string;
  error?: string | null;
  top4: TopHit[];
};

export type PreprocessInfo = {
  resize_short_edge: number;
  center_crop: number;
  imagenet_mean: number[];
  imagenet_std: number[];
  layout: string;
  temperature_baked: boolean;
  temperature: number;
  note: string;
};

export type PredictResponse = {
  model: string;
  provider: string;
  preprocessing: PreprocessInfo;
  results: PredictItem[];
};

export type MetaResponse = {
  model: string;
  num_classes: number;
  provider: string;
  img_size: number;
  imagenet_mean: number[];
  imagenet_std: number[];
  temperature_baked: boolean;
  temperature: number;
  preprocessing: PreprocessInfo;
};

export type QueuedFile = {
  id: string;
  file: File;
  url: string;
};
