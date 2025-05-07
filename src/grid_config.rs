//! This module implements code for configuring a crossword-filling operation, independent of the
//! specific fill algorithm.

use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde_derive::{Deserialize, Serialize};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::types::{GlyphId, WordId};
use crate::util::build_glyph_counts_by_cell;
use crate::word_list::WordList;

/// An identifier for the intersection between two slots; these correspond one-to-one with checked
/// squares in the grid and are used to track weights (i.e., how often each square is involved in
/// a domain wipeout).
pub type CrossingId = usize;

/// An identifier for a given slot, based on its index in the `GridConfig`'s `slot_configs` field.
pub type SlotId = usize;

/// Zero-indexed x and y coords for a cell in the grid, where y = 0 in the top row.
pub type GridCoord = (usize, usize);

/// The direction that a slot is facing.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[allow(dead_code)]
pub enum Direction {
    Across,
    Down,
}

/// A struct representing a crossing between one slot and another, referencing the other slot's id
/// and the location of the intersection within the other slot.
#[derive(Debug, Clone)]
pub struct Crossing {
    pub other_slot_id: SlotId,
    pub other_slot_cell: usize,
    pub crossing_id: CrossingId,
}

/// A struct representing the aspects of a slot in the grid that are static during filling.
#[derive(Debug, Clone)]
pub struct SlotConfig {
    pub id: SlotId,
    pub start_cell: GridCoord,
    pub direction: Direction,
    pub length: usize,
    pub crossings: Vec<Option<Crossing>>,
    pub min_score_override: Option<u16>,
    pub filter_pattern: Option<Regex>,
    pub number: usize, // Added to store the crossword number for this slot
}

impl SlotConfig {
    /// Generate the coords for each cell of this slot.
    #[must_use]
    pub fn cell_coords(&self) -> Vec<GridCoord> {
        (0..self.length)
            .map(|cell_idx| match self.direction {
                Direction::Across => (self.start_cell.0 + cell_idx, self.start_cell.1),
                Direction::Down => (self.start_cell.0, self.start_cell.1 + cell_idx),
            })
            .collect()
    }

    /// Generate the indices of this slot's cells in a flat fill array like `GridConfig.fill`.
    #[must_use]
    pub fn cell_fill_indices(&self, grid_width: usize) -> Vec<usize> {
        self.cell_coords()
            .iter()
            .map(|loc| loc.0 + loc.1 * grid_width)
            .collect()
    }

    /// Get the values of this slot's cells in a flat fill array like `GridConfig.fill`.
    #[must_use]
    pub fn fill(&self, fill: &[Option<GlyphId>], grid_width: usize) -> Vec<Option<GlyphId>> {
        self.cell_fill_indices(grid_width)
            .iter()
            .map(|&idx| {
                if idx < fill.len() { // Ensure index is within bounds
                    fill[idx]
                } else {
                    None // Or handle error appropriately
                }
            })
            .collect()
    }

    /// Get this slot's `fill` if and only if all of its cells are populated.
    #[must_use]
    pub fn complete_fill(
        &self,
        fill: &[Option<GlyphId>],
        grid_width: usize,
    ) -> Option<Vec<GlyphId>> {
        self.fill(fill, grid_width).into_iter().collect()
    }

    /// Generate a `SlotSpec` identifying this slot.
    #[must_use]
    pub fn slot_spec(&self) -> SlotSpec {
        SlotSpec {
            start_cell: self.start_cell,
            direction: self.direction,
            length: self.length,
            // Number is not part of SlotSpec by default, handled separately if needed for keying
        }
    }

    /// Generate a string key identifying this slot.
    #[must_use]
    pub fn slot_key(&self) -> String {
        self.slot_spec().to_key()
    }
}

/// A struct holding references to all of the information needed as input to a crossword filling
/// operation.
#[allow(dead_code)]
#[derive(Clone)]
pub struct GridConfig<'a> {
    /// The word list used to fill the grid; see `word_list.rs`.
    pub word_list: &'a WordList,

    /// A flat array of letters filled into the grid, in order of row and then column. `None` can
    /// represent a block or an unfilled cell.
    pub fill: &'a [Option<GlyphId>],

    /// Config representing all of the slots in the grid and their crossings.
    pub slot_configs: &'a [SlotConfig],

    /// An array of available words for each (respective) slot, based on both the word list config
    /// and the existing letters filled into the grid.
    pub slot_options: &'a [Vec<WordId>],

    /// The width and height of the grid.
    pub width: usize,
    pub height: usize,

    /// The number of distinct crossings represented in all of the `slot_configs`.
    pub crossing_count: usize,

    /// An optional atomic flag that can be set to signal that the fill operation should be canceled.
    pub abort: Option<&'a AtomicBool>,

    /// Map from (number, direction) to SlotId
    pub slot_map: &'a HashMap<(usize, Direction), SlotId>,
}

impl<'a> GridConfig<'a> {
    /// Retrieves the SlotId for a given slot number and direction.
    /// Slot numbers are typically 1-indexed.
    pub fn slot_id_for_number_and_dir(&self, number: usize, dir: Direction) -> Option<SlotId> {
        self.slot_map.get(&(number, dir)).copied()
    }
}


/// A struct that owns a copy of each piece of information needed by `GridConfig`.
pub struct OwnedGridConfig {
    pub word_list: WordList,
    pub fill: Vec<Option<GlyphId>>,
    pub slot_configs: Vec<SlotConfig>,
    pub slot_options: Vec<Vec<WordId>>,
    pub width: usize,
    pub height: usize,
    pub crossing_count: usize,
    pub abort: Option<Arc<AtomicBool>>,
    pub slot_map: HashMap<(usize, Direction), SlotId>, // Added
}

impl OwnedGridConfig {
    #[allow(dead_code)]
    #[must_use]
    pub fn to_config_ref(&self) -> GridConfig {
        GridConfig {
            word_list: &self.word_list,
            fill: &self.fill,
            slot_configs: &self.slot_configs,
            slot_options: &self.slot_options,
            width: self.width,
            height: self.height,
            crossing_count: self.crossing_count,
            abort: self.abort.as_deref(),
            slot_map: &self.slot_map,
        }
    }
}

/// Given a configured grid, reorder the options for each slot so that the "best" choices are at the
/// front. This is a balance between fillability (the most important factor, since our odds of being
/// able to find a fill in a reasonable amount of time depend on how many tries it takes us to find
/// a usable word for each slot) and quality metrics like word score and letter score.
#[allow(clippy::cast_lossless)]
pub fn sort_slot_options(
    word_list: &WordList,
    slot_configs: &[SlotConfig],
    slot_options: &mut [Vec<WordId>],
) {
    // To calculate the fillability score for each word, we need statistics about which letters are
    // most likely to appear in each position for each slot.
    let glyph_counts_by_cell_by_slot: Vec<_> = slot_configs
        .iter()
        .map(|slot_config| {
            build_glyph_counts_by_cell(word_list, slot_config.length, &slot_options[slot_config.id])
        })
        .collect();

    // Now we can actually sort the options.
    for slot_idx in 0..slot_configs.len() {
        let slot_config = &slot_configs[slot_idx];
        let current_slot_options = &mut slot_options[slot_idx]; // Changed variable name

        current_slot_options.sort_by_cached_key(|&option| { // Changed variable name
            let word = &word_list.words[slot_config.length][option];

            // To calculate the fill score for a word, average the logarithms of the number of
            // crossing options that are compatible with each letter (based on the grid geometry).
            // This is kind of arbitrary, but it seems like it makes sense because we care a lot
            // more about the difference between 1 option and 5 options or 5 options and 20 options
            // than 100 options and 500 options.
            let fill_score = slot_config
                .crossings
                .iter()
                .zip(&word.glyphs)
                .map(|(crossing, &glyph)| match crossing {
                    Some(crossing) => {
                        let crossing_counts_by_cell =
                            &glyph_counts_by_cell_by_slot[crossing.other_slot_id];

                        (crossing_counts_by_cell[crossing.other_slot_cell][glyph] as f32).log10()
                    }
                    None => 0.0,
                })
                .fold(0.0, |a, b| a + b)
                / (slot_config.length as f32);

            // This is arbitrary, based on visual inspection of the ranges for each value. Generally
            // increasing the weight of `fill_score` relative to the other two will reduce fill
            // time.
            -((fill_score * 900.0) as i64
                + ((word.letter_score as f32) * 5.0) as i64
                + ((word.score as f32) * 5.0) as i64)
        });
    }
}

/// A struct identifying a specific slot in the grid.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SlotSpec {
    pub start_cell: GridCoord,
    pub direction: Direction,
    pub length: usize,
    // pub number: Option<usize>, // Keep SlotSpec simpler, handle numbering separately
}

impl SlotSpec {
    /// Parse a string like "1,2,down,5" into a `SlotSpec` struct.
    pub fn from_key(key: &str) -> Result<SlotSpec, String> {
        let key_parts: Vec<&str> = key.split(',').collect();
        if key_parts.len() != 4 {
            return Err(format!("invalid slot key: {key}"));
        }

        let x: Result<usize, _> = key_parts[0].parse();
        let y: Result<usize, _> = key_parts[1].parse();
        let direction: Option<Direction> = match key_parts[2] {
            "across" => Some(Direction::Across),
            "down" => Some(Direction::Down),
            _ => None,
        };
        let length: Result<usize, _> = key_parts[3].parse();

        if let (Ok(x), Ok(y), Some(direction), Ok(length)) = (x, y, direction, length) {
            Ok(SlotSpec {
                start_cell: (x, y),
                direction,
                length,
            })
        } else {
            Err(format!("invalid slot key: {key:?}"))
        }
    }

    /// Represent this slot as a string like "1,2,down,5".
    #[must_use]
    pub fn to_key(&self) -> String {
        let direction_str = match self.direction { // Renamed variable
            Direction::Across => "across",
            Direction::Down => "down",
        };
        format!(
            "{},{},{},{}",
            self.start_cell.0, self.start_cell.1, direction_str, self.length, // Used renamed variable
        )
    }

    /// Does this spec match the given slot config?
    #[must_use]
    pub fn matches_slot(&self, slot: &SlotConfig) -> bool {
        self.start_cell == slot.start_cell
            && self.direction == slot.direction
            && self.length == slot.length
    }

    /// Generate the coords for each cell of this entry.
    #[must_use]
    pub fn cell_coords(&self) -> Vec<GridCoord> {
        (0..self.length)
            .map(|cell_idx| match self.direction {
                Direction::Across => (self.start_cell.0 + cell_idx, self.start_cell.1),
                Direction::Down => (self.start_cell.0, self.start_cell.1 + cell_idx),
            })
            .collect()
    }
}

/// Serialize a `SlotSpec` into a string key.
#[cfg(feature = "serde")]
impl Serialize for SlotSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_key())
    }
}

/// Deserialize a `SlotSpec` from a string key.
#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for SlotSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw_string = String::deserialize(deserializer)?;
        SlotSpec::from_key(&raw_string).map_err(serde::de::Error::custom)
    }
}


/// Given `SlotSpec` structs specifying the positions of the slots in a grid, generate
/// `SlotConfig`s containing derived information about crossings, etc.
/// Also generates a map from (number, direction) to SlotId.
#[must_use]
pub fn generate_slot_configs_and_map(
    entries: &[SlotSpec], // These are the identified slots (e.g. from template parsing)
    grid_width: usize, 
    grid_height: usize, 
    is_block_fn: &impl Fn(usize, usize) -> bool, // Function to check if a cell is a block
) -> (Vec<SlotConfig>, usize, HashMap<(usize, Direction), SlotId>) {
    
    let mut slot_configs: Vec<SlotConfig> = vec![];
    let mut slot_map: HashMap<(usize, Direction), SlotId> = HashMap::new();

    // Build a map from cell location to (list of (entry_idx, cell_idx_in_entry))
    let mut cell_to_entry_parts: HashMap<GridCoord, Vec<(usize, usize)>> = HashMap::new();
    for (entry_idx, entry) in entries.iter().enumerate() {
        for (cell_idx_in_entry, &loc) in entry.cell_coords().iter().enumerate() {
            cell_to_entry_parts.entry(loc).or_default().push((entry_idx, cell_idx_in_entry));
        }
    }
    
    // Assign numbers to cells that start slots
    // Numbering logic: iterate through grid cells, assign a new number if a slot starts there.
    let mut current_clue_number = 0;
    let mut coord_to_number: HashMap<GridCoord, usize> = HashMap::new();
    // Iterate in standard reading order (top-to-bottom, left-to-right)
    for r in 0..grid_height {
        for c in 0..grid_width {
            if is_block_fn(c,r) {
                continue;
            }
            let starts_across = (c == 0 || is_block_fn(c - 1, r)) && (c + 1 < grid_width && !is_block_fn(c + 1, r));
            let starts_down = (r == 0 || is_block_fn(c, r - 1)) && (r + 1 < grid_height && !is_block_fn(c, r + 1));

            if starts_across || starts_down {
                current_clue_number += 1;
                coord_to_number.insert((c,r), current_clue_number);
            }
        }
    }
    
    let mut constraint_id_cache: Vec<(SlotId, SlotId)> = vec![];

    for (entry_idx, entry_spec) in entries.iter().enumerate() {
        let crossings: Vec<Option<Crossing>> = entry_spec
            .cell_coords()
            .iter()
            .map(|&loc| {
                let entry_parts_in_cell = cell_to_entry_parts.get(&loc).unwrap(); // Should always exist
                let mut other_entry_part_option: Option<(usize, usize)> = None;

                for &(other_entry_idx, other_cell_idx_in_entry) in entry_parts_in_cell {
                    if other_entry_idx != entry_idx { // Found a crossing entry
                        other_entry_part_option = Some((other_entry_idx, other_cell_idx_in_entry));
                        break; 
                    }
                }
                
                if let Some((other_slot_id, other_slot_cell)) = other_entry_part_option {
                    // Check cache for existing constraint ID
                    let crossing_id = if let Some(found_constraint_id) = constraint_id_cache
                        .iter()
                        .enumerate()
                        .find(|&(_, &id_pair)| {
                            (id_pair.0 == entry_idx && id_pair.1 == other_slot_id) ||
                            (id_pair.0 == other_slot_id && id_pair.1 == entry_idx)
                        })
                        .map(|(id, _)| id)
                    {
                        found_constraint_id
                    } else {
                        // Add new constraint ID to cache
                        constraint_id_cache.push((entry_idx, other_slot_id));
                        constraint_id_cache.len() - 1
                    };

                    Some(Crossing {
                        other_slot_id, // This is the SlotId of the crossing slot
                        other_slot_cell, // This is the cell index within the crossing slot
                        crossing_id,
                    })
                } else {
                    None // No other entry crosses this cell of the current entry_spec
                }
            })
            .collect();

        let slot_clue_number = coord_to_number.get(&entry_spec.start_cell)
            .copied()
            .unwrap_or_else(|| panic!("Slot at {:?} direction {:?} did not get a number. entries: {:?}", entry_spec.start_cell, entry_spec.direction, entries));


        slot_map.insert((slot_clue_number, entry_spec.direction), entry_idx);
        slot_configs.push(SlotConfig {
            id: entry_idx,
            start_cell: entry_spec.start_cell,
            direction: entry_spec.direction,
            length: entry_spec.length,
            crossings,
            min_score_override: None,
            filter_pattern: None,
            number: slot_clue_number,
        });
    }

    (slot_configs, constraint_id_cache.len(), slot_map)
}


/// Given a single slot's fill, minimum score, and optional filter pattern, generate the possible
/// options for that slot by starting with the complete word list and then removing words that
/// contradict the criteria. If `allowed_word_ids` is provided, the given words will be included in
/// the options as long as they don't contradict the fill, regardless of whether they match the min
/// score and filter pattern.
pub fn generate_slot_options(
    word_list: &mut WordList, // Made mutable as get_word_id_or_add_hidden can modify it
    entry_fill: &[Option<GlyphId>],
    min_score: u16,
    filter_pattern: Option<&Regex>,
    allowed_word_ids: Option<&HashSet<WordId>>,
) -> Vec<WordId> {
    let length = entry_fill.len();

    if length == 0 || word_list.words.len() <= length { 
        return vec![];
    }

    let complete_fill: Option<Vec<GlyphId>> = entry_fill.iter().copied().collect();

    if let Some(filled_glyphs) = complete_fill { // Renamed variable
        let word_string: String = filled_glyphs // Used renamed variable
            .iter()
            .map(|&glyph_id| word_list.glyphs[glyph_id])
            .collect();

        let (_word_length, word_id) = word_list.get_word_id_or_add_hidden(&word_string);

        vec![word_id]
    } else {
        let options: Vec<WordId> = (0..word_list.words[length].len())
            .filter(|&word_id| {
                let word = &word_list.words[length][word_id];
                let enforce_criteria = allowed_word_ids.map_or(true, |allowed| {
                    !allowed.contains(&word_id)
                });

                if enforce_criteria {
                    if word.hidden || word.score < min_score {
                        return false;
                    }

                    if let Some(pattern) = filter_pattern.as_ref() {
                        if !pattern
                            .is_match(&word.normalized_string)
                            .unwrap_or(false)
                        {
                            return false;
                        }
                    }
                }

                entry_fill.iter().enumerate().all(|(cell_idx, cell_fill_glyph)| {
                    cell_fill_glyph
                        .map(|g| word.glyphs.get(cell_idx).map_or(false, |&wg| g == wg)) // Added bounds check for word.glyphs
                        .unwrap_or(true)
                })
            })
            .collect();

        options
    }
}

/// Given an input fill and an array of slot configs, generate the possible options for each slot
/// by starting with the complete word list and then removing words that contradict any fill that's
/// already present in the grid or violate criteria like minimum score or filter pattern.
pub fn generate_all_slot_options(
    word_list: &mut WordList, // Made mutable
    fill: &[Option<GlyphId>],
    slot_configs: &[SlotConfig],
    grid_width: usize,
    global_min_score: u16,
) -> Vec<Vec<WordId>> {
    slot_configs
        .iter()
        .map(|slot| {
            generate_slot_options(
                word_list, // Pass mutable word_list
                &slot.fill(fill, grid_width),
                slot.min_score_override.unwrap_or(global_min_score),
                slot.filter_pattern.as_ref(),
                None,
            )
        })
        .collect()
}

/// Generate an `OwnedGridConfig` representing a grid with specified entries.
#[must_use]
pub fn generate_grid_config(
    mut word_list: WordList, // Takes ownership and can be mutated
    entries: &[SlotSpec],
    raw_char_fill: &[Option<String>], // The character grid ('.', '#', 'a', etc.)
    width: usize,
    height: usize,
    min_score: u16,
) -> OwnedGridConfig {
    let is_block_fn = |x: usize, y: usize| -> bool {
        if y >= height || x >= width { return true; } 
        let idx = y * width + x;
        raw_char_fill.get(idx).and_then(|c| c.as_ref()).map_or(false, |s| s == "#")
    };

    let (slot_configs, crossing_count, slot_map) =
        generate_slot_configs_and_map(entries, width, height, &is_block_fn);

    let fill_glyphs: Vec<Option<GlyphId>> = raw_char_fill
        .iter()
        .map(|cell_str_opt| {
            cell_str_opt
                .as_ref()
                .and_then(|cell_str| {
                    if cell_str == "#" || cell_str == "." { 
                        None
                    } else {
                        // Ensure word_list is mutable here for glyph_id_for_char
                        Some(word_list.glyph_id_for_char(cell_str.chars().next().unwrap()))
                    }
                })
        })
        .collect();

    let mut slot_options =
        generate_all_slot_options(&mut word_list, &fill_glyphs, &slot_configs, width, min_score);

    sort_slot_options(&word_list, &slot_configs, &mut slot_options); // word_list is & here, which is fine

    OwnedGridConfig {
        word_list, // word_list is moved here
        fill: fill_glyphs,
        slot_configs,
        slot_options,
        width,
        height,
        crossing_count,
        abort: None,
        slot_map,
    }
}


/// Generate a list of `SlotSpec`s from a template string with . representing empty cells, # representing
/// blocks, and letters representing themselves.
#[allow(dead_code)]
#[must_use]
pub fn generate_slots_from_template_string(template_str: &str, _width: usize, _height: usize) -> Vec<SlotSpec> { // Underscore unused width/height
    let template_chars: Vec<Vec<char>> = template_str
        .lines()
        .map(str::trim) // Trim lines before collecting
        .filter(|l| !l.is_empty()) // Filter empty lines
        .map(|line| line.chars().collect())
        .collect();

    // Ensure consistent grid dimensions based on parsed template_chars
    let actual_height = template_chars.len();
    let actual_width = if actual_height > 0 { template_chars[0].len() } else { 0 };

    if actual_height == 0 || actual_width == 0 { return vec![]; }


    let is_block_char = |x: usize, y: usize, template: &Vec<Vec<char>>| -> bool {
        if y >= actual_height || x >= actual_width { return true; } // out of bounds is like a block
        template[y][x] == '#'
    };
    
    let mut slot_specs: Vec<SlotSpec> = vec![];

    // Across slots
    for r_idx in 0..actual_height { // Use actual_height
        let mut c_idx = 0;
        while c_idx < actual_width { // Use actual_width
            if !is_block_char(c_idx, r_idx, &template_chars) {
                let start_c = c_idx;
                while c_idx < actual_width && !is_block_char(c_idx, r_idx, &template_chars) {
                    c_idx += 1;
                }
                let length = c_idx - start_c;
                if length > 1 { 
                    slot_specs.push(SlotSpec {
                        start_cell: (start_c, r_idx),
                        length,
                        direction: Direction::Across,
                    });
                }
            } else {
                c_idx += 1;
            }
        }
    }

    // Down slots
    for c_idx in 0..actual_width { // Use actual_width
        let mut r_idx = 0;
        while r_idx < actual_height { // Use actual_height
            if !is_block_char(c_idx, r_idx, &template_chars) {
                let start_r = r_idx;
                while r_idx < actual_height && !is_block_char(c_idx, r_idx, &template_chars) {
                    r_idx += 1;
                }
                let length = r_idx - start_r;
                if length > 1 { 
                     slot_specs.push(SlotSpec {
                        start_cell: (c_idx, start_r),
                        length,
                        direction: Direction::Down,
                    });
                }
            } else {
                r_idx += 1;
            }
        }
    }
    
    slot_specs.sort_by_key(|s| (s.direction, s.start_cell.1, s.start_cell.0));
    slot_specs
}


/// Generate an `OwnedGridConfig` from a template string with . representing empty cells, # representing
/// blocks, and letters representing themselves.
#[allow(dead_code)]
#[must_use]
pub fn generate_grid_config_from_template_string(
    word_list: WordList, // Takes ownership
    template: &str, 
    min_score: u16,
) -> OwnedGridConfig {

    let lines: Vec<&str> = template.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let height = lines.len();
    let width = lines.first().map_or(0, |l| l.chars().count());

    if height == 0 || width == 0 {
        // Return a default or error OwnedGridConfig if template is empty
        return OwnedGridConfig {
            word_list, // still pass ownership
            fill: vec![],
            slot_configs: vec![],
            slot_options: vec![],
            width: 0,
            height: 0,
            crossing_count: 0,
            abort: None,
            slot_map: HashMap::new(),
        };
    }

    let slot_specs = generate_slots_from_template_string(template, width, height);

    let raw_char_fill: Vec<Option<String>> = lines // Use processed lines
        .into_iter() // Consuming iterator
        .flat_map(|line| {
            line.chars()
                .map(|c| {
                    if c == '#' { 
                        Some("#".to_string())
                    } else if c == '.' { 
                        None
                    }
                     else { 
                        Some(c.to_lowercase().to_string())
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    generate_grid_config(
        word_list,
        &slot_specs,
        &raw_char_fill,
        width,
        height,
        min_score,
    )
}


/// A struct recording a slot assignment made during a fill process.
#[derive(Debug, Clone)]
pub struct Choice {
    pub slot_id: SlotId,
    pub word_id: WordId,
}

/// Turn the given grid config and fill choices into a rendered string.
#[allow(dead_code)]
#[must_use]
pub fn render_grid(config: &GridConfig, choices: &[Choice]) -> String {
    if config.height == 0 || config.width == 0 {
        return String::from("Empty grid");
    }
    let mut grid_chars: Vec<Vec<Option<char>>> = (0..config.height)
        .map(|_| (0..config.width).map(|_| None).collect::<Vec<_>>())
        .collect();

    // Mark blocks first based on where letters *cannot* go (i.e. not part of any slot)
    let mut is_fillable_cell = vec![vec![false; config.width]; config.height];
    for slot_config in config.slot_configs.iter() {
        for &(x,y) in &slot_config.cell_coords() {
            if y < config.height && x < config.width {
                is_fillable_cell[y][x] = true;
            }
        }
    }
    for r_idx in 0..config.height {
        for c_idx in 0..config.width {
            if !is_fillable_cell[r_idx][c_idx] {
                grid_chars[r_idx][c_idx] = Some('#');
            }
        }
    }

    // Place pre-filled characters from initial grid template
    for r_idx in 0..config.height {
        for c_idx in 0..config.width {
            let fill_idx = r_idx * config.width + c_idx;
            if fill_idx < config.fill.len() { // Bounds check
                if let Some(glyph_id) = config.fill[fill_idx] {
                     if grid_chars[r_idx][c_idx].is_none() { // Don't overwrite if already marked as block
                        grid_chars[r_idx][c_idx] = Some(config.word_list.glyphs[glyph_id]);
                     }
                }
            }
        }
    }
    
    // Overlay choices from the solver
    for &Choice { slot_id, word_id } in choices {
        let slot_config = &config.slot_configs[slot_id];
        // Ensure word_id is valid for the slot_config.length
        if slot_config.length == 0 || config.word_list.words.len() <= slot_config.length || config.word_list.words[slot_config.length].len() <= word_id {
            // Skip this choice if word_id is invalid, or log an error
            // eprintln!("Warning: Invalid word_id {} for slot {} with length {}", word_id, slot_id, slot_config.length);
            continue;
        }
        let word = &config.word_list.words[slot_config.length][word_id];

        for (cell_idx_in_word, &glyph) in word.glyphs.iter().enumerate() {
            let (x, y) = match slot_config.direction {
                Direction::Across => (
                    slot_config.start_cell.0 + cell_idx_in_word,
                    slot_config.start_cell.1,
                ),
                Direction::Down => (
                    slot_config.start_cell.0,
                    slot_config.start_cell.1 + cell_idx_in_word,
                ),
            };
            if y < config.height && x < config.width {
                 grid_chars[y][x] = Some(config.word_list.glyphs[glyph]);
            }
        }
    }

    grid_chars.iter()
        .map(|line| {
            line.iter()
                .map(|cell| cell.unwrap_or('.').to_string()) 
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}


#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use crate::grid_config::{Direction, SlotSpec};

    #[test]
    fn test_slot_spec_serialization() {
        let slot_spec = SlotSpec {
            start_cell: (1, 2),
            direction: Direction::Across,
            length: 5,
        };

        let slot_key = serde_json::to_string(&slot_spec).unwrap();

        assert_eq!(slot_key, "\"1,2,across,5\"");
    }

    #[test]
    fn test_slot_spec_deserialization() {
        let slot_spec: SlotSpec = serde_json::from_str("\"3,4,down,12\"").unwrap();

        assert_eq!(
            slot_spec,
            SlotSpec {
                start_cell: (3, 4),
                direction: Direction::Down,
                length: 12,
            }
        );
    }
}

