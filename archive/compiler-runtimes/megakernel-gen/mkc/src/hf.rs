//! HuggingFace ingestion: `config.json` + every `*.safetensors` header in a
//! model directory, resolved into a name -> byte-slice index.
//!
//! Nothing here knows about any particular architecture.  It answers two
//! questions: what does the config say, and where does each tensor live.

use crate::ir::{DType, Slab};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct HfModel {
    #[allow(dead_code)] // kept for error messages and future relative-path resolution
    pub dir: PathBuf,
    pub config: Value,
    pub gen_config: Option<Value>,
    /// safetensors files, sorted; index into this is `Slab::file`
    pub files: Vec<String>,
    /// tensor name -> where its bytes are
    pub tensors: BTreeMap<String, Slab>,
}

fn read_st_header(path: &Path, file_idx: usize) -> Result<Vec<(String, Slab)>, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)
        .map_err(|e| format!("{}: short header: {}", path.display(), e))?;
    let hlen = u64::from_le_bytes(len8);
    if hlen > 512 * 1024 * 1024 {
        return Err(format!("{}: implausible header length {}", path.display(), hlen));
    }
    let mut buf = vec![0u8; hlen as usize];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{}: truncated header: {}", path.display(), e))?;
    let json: Value = serde_json::from_slice(&buf)
        .map_err(|e| format!("{}: bad header json: {}", path.display(), e))?;
    let obj = json.as_object().ok_or("header is not an object")?;
    let data_start = 8 + hlen;
    let mut out = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let m = meta.as_object().ok_or("tensor meta is not an object")?;
        let dt = m
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or("tensor meta has no dtype")?;
        let dtype = DType::from_st(dt).ok_or_else(|| format!("unsupported dtype {}", dt))?;
        let shape: Vec<usize> = m
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or("tensor meta has no shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = m
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or("tensor meta has no data_offsets")?;
        let s = off[0].as_u64().unwrap_or(0);
        let e = off[1].as_u64().unwrap_or(0);
        out.push((
            name.clone(),
            Slab {
                file: file_idx,
                offset: data_start + s,
                nbytes: e - s,
                dtype,
                shape,
                name: name.clone(),
            },
        ));
    }
    Ok(out)
}

impl HfModel {
    pub fn load(dir: &Path) -> Result<HfModel, String> {
        let cfg_path = dir.join("config.json");
        let cfg_txt = fs::read_to_string(&cfg_path)
            .map_err(|e| format!("{}: {}", cfg_path.display(), e))?;
        let config: Value =
            serde_json::from_str(&cfg_txt).map_err(|e| format!("config.json: {}", e))?;

        let gen_config = fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());

        let mut files: Vec<String> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {}", dir.display(), e))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors in {}", dir.display()));
        }

        let mut tensors = BTreeMap::new();
        for (i, fname) in files.iter().enumerate() {
            for (n, s) in read_st_header(&dir.join(fname), i)? {
                tensors.insert(n, s);
            }
        }
        Ok(HfModel {
            dir: dir.to_path_buf(),
            config,
            gen_config,
            files,
            tensors,
        })
    }

    // ---- config accessors: tolerant of the nesting HF uses for multimodal --

    /// The text-model sub-config if there is one, else the root.
    pub fn text_cfg(&self) -> &Value {
        for k in ["text_config", "llm_config", "language_config"] {
            if let Some(v) = self.config.get(k) {
                if v.is_object() {
                    return v;
                }
            }
        }
        &self.config
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        let t = self.text_cfg();
        t.get(key).or_else(|| self.config.get(key))
    }

    pub fn usize_of(&self, key: &str) -> Option<usize> {
        self.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
    }
    pub fn f32_of(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
    }
    pub fn bool_of(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }
    pub fn str_of(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    pub fn arch(&self) -> String {
        if let Some(a) = self
            .config
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
        {
            return a.to_string();
        }
        self.str_of("model_type").unwrap_or_else(|| "unknown".into())
    }

    pub fn model_type(&self) -> String {
        self.str_of("model_type").unwrap_or_else(|| "unknown".into())
    }

    /// First tensor whose name matches any of the candidates.
    pub fn any(&self, cands: &[String]) -> Option<&Slab> {
        cands.iter().find_map(|c| self.tensors.get(c))
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for src in [self.gen_config.as_ref(), Some(&self.config)] {
            let Some(v) = src else { continue };
            if let Some(e) = v.get("eos_token_id") {
                match e {
                    Value::Number(n) => out.push(n.as_u64().unwrap_or(0) as u32),
                    Value::Array(a) => {
                        for x in a {
                            if let Some(n) = x.as_u64() {
                                out.push(n as u32)
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !out.is_empty() {
                break;
            }
        }
        out.sort();
        out.dedup();
        out
    }

    pub fn bos_id(&self) -> Option<u32> {
        for src in [self.gen_config.as_ref(), Some(&self.config)] {
            let Some(v) = src else { continue };
            if let Some(n) = v.get("bos_token_id").and_then(|x| x.as_u64()) {
                return Some(n as u32);
            }
        }
        None
    }
}
