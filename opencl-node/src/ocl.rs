use crate::lru_cache::LruCache;
use crate::map::MapCl;
use crate::mini_lru_cache::MiniLruCache;

use io::Error as IoError;
use opencl_collections::cache::config::CacheSrc;
use opencl_collections::cache::handle::lru::LRUCacheHandle;
use opencl_collections::cache::handle::mini_lru::MiniLRUCacheHandle;
use opencl_collections::config::DEFAULT_DEVICE_INDEX;
use opencl_collections::map::config::{MapSrc, MAX_FIND_WORK_SIZE};
use opencl_collections::map::handle::{Handle, MapHandle};
use opencl_collections::opencl::wrapper::system::System;
use std::io;
use std::io::ErrorKind;
use std::sync::Arc;

#[derive(Debug)]
#[napi(object)]
pub struct MapBlockConfig {
  pub value_len: i32,
  pub capacity: i32,
}

#[derive(Debug)]
#[napi(object)]
pub struct MapOptions {
  pub total_maps: Option<i32>,
  pub blocks: Vec<MapBlockConfig>,
}

#[derive(Debug)]
#[napi(object)]
pub struct MiniLRUCacheConfig {
  pub capacity: i32,
}

#[derive(Debug)]
#[napi(object)]
pub struct LRUCacheConfig {
  pub value_len: i32,
  pub capacity: i32,
}

#[derive(Debug)]
#[napi(object)]
pub struct CacheOptions {
  pub mini_lru: Option<Vec<MiniLRUCacheConfig>>,
  pub lru: Option<Vec<LRUCacheConfig>>,
}

#[derive(Debug)]
#[napi(object)]
pub struct Collections {
  pub map: Option<MapOptions>,
  pub cache: Option<CacheOptions>,
}

#[derive(Debug)]
#[napi(object)]
pub struct MapClOptions {
  pub map_id: i32,
}

#[napi]
pub struct Device {
  cache_src: CacheSrc,
  cache_ocl_block: Option<Arc<System>>,
  map_src: MapSrc<i32>,
  map_ocl_block: Option<Arc<System>>,
}

impl Device {
  pub fn create(collections: Collections) -> io::Result<Self> {
    let mut map_src = MapSrc::new(1);

    let mut cache_src = CacheSrc::new();

    // println!("collections {:#?}", collections);

    if let Some(map_options) = collections.map {
      if let Some(n) = map_options.total_maps {
        map_src.set_total_maps(n as usize);
      }

      for c in map_options.blocks {
        map_src.add(c.value_len as usize, c.capacity as usize)
      }
    }

    if let Some(cache_options) = collections.cache {
      if let Some(configs) = cache_options.mini_lru {
        for c in configs {
          cache_src.add_mini_lru(c.capacity as usize)
        }
      }

      if let Some(configs) = cache_options.lru {
        for c in configs {
          cache_src.add_lru(256, c.value_len as usize, c.capacity as usize)
        }
      }
    }

    // initialize
    let mut map_ocl_block = None;

    if !map_src.get_configs().is_empty() {
      map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);
      map_src.add_map_get_program_src(MAX_FIND_WORK_SIZE);

      let ocl_block = System::new(DEFAULT_DEVICE_INDEX, &map_src.build())?;
      let arc_ocl_block = Arc::new(ocl_block);

      let map_handle = Handle::new(&map_src, arc_ocl_block.clone());
      map_handle.initialize_all_maps()?;

      map_ocl_block = Some(arc_ocl_block);
    }

    let mut cache_ocl_block = None;

    if !cache_src.is_empty() {
      let ocl_block = System::new(DEFAULT_DEVICE_INDEX, &cache_src.build())?;
      let arc_ocl_block = Arc::new(ocl_block);

      cache_ocl_block = Some(arc_ocl_block);
    }

    Ok(Self {
      cache_src,
      cache_ocl_block,
      map_src,
      map_ocl_block,
    })
  }

  pub fn _create_map(&mut self, options: MapClOptions) -> io::Result<MapCl> {
    match &self.map_ocl_block {
      None => Err(IoError::new(ErrorKind::Other, "map src empty")),
      Some(map_ocl_block) => {
        let map_base = MapHandle::new(
          options.map_id as usize,
          &self.map_src,
          map_ocl_block.clone(),
        );
        Ok(MapCl::new(map_base))
      }
    }
  }

  pub fn _create_mini_lru_cache(&self, id: i32) -> io::Result<MiniLruCache> {
    match &self.cache_ocl_block {
      None => Err(IoError::new(ErrorKind::Other, "cache src empty")),
      Some(cache_ocl_block) => {
        let config = self.cache_src.get_mini_lru_by_id(id as usize)?;
        let cache_base = MiniLRUCacheHandle::new(config, cache_ocl_block.clone());
        cache_base.initialize()?;

        Ok(MiniLruCache::new(cache_base))
      }
    }
  }

  pub fn _create_lru_cache(&self, id: i32) -> io::Result<LruCache> {
    match &self.cache_ocl_block {
      None => Err(IoError::new(ErrorKind::Other, "cache src empty")),
      Some(cache_ocl_block) => {
        let config = self.cache_src.get_lru_by_id(id as usize)?;
        let cache_base = LRUCacheHandle::new(config, cache_ocl_block.clone());
        cache_base.initialize()?;

        Ok(LruCache::new(cache_base))
      }
    }
  }
}

#[napi]
impl Device {
  #[napi(constructor)]
  pub fn new(collections: Collections) -> napi::Result<Self> {
    let d = Self::create(collections)?;
    Ok(d)
  }

  #[napi]
  pub fn print_map_src(&self) -> napi::Result<()> {
    println!("{:#?}", self.map_src);
    Ok(())
  }

  #[napi]
  pub fn print_cache_src(&self) -> napi::Result<()> {
    println!("{:#?}", self.cache_src);
    Ok(())
  }

  #[napi]
  pub fn create_map(&mut self, options: MapClOptions) -> napi::Result<MapCl> {
    let m = self._create_map(options)?;
    Ok(m)
  }

  #[napi]
  pub fn create_mini_lru_cache(&self, id: i32) -> napi::Result<MiniLruCache> {
    let c = self._create_mini_lru_cache(id)?;
    Ok(c)
  }

  #[napi]
  pub fn create_lru_cache(&self, id: i32) -> napi::Result<LruCache> {
    let c = self._create_lru_cache(id)?;
    Ok(c)
  }
}
