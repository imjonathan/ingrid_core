use clap::Parser; 
use ingrid_core::backtracking_search::find_fill;
use ingrid_core::grid_config::{
    generate_grid_config_from_template_string, render_grid, Direction, SlotId // Removed generate_slot_options
};
// Add WordId import
use ingrid_core::types::WordId; 
use ingrid_core::word_list::{WordList, WordListSourceConfig, WordListError as IngridWordListError, Word as IngridWord}; 
use std::collections::HashMap; 
use std::fmt::{Debug, Formatter};
use std::fs;
use std::path::PathBuf; 
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

const STWL_RAW: &str = include_str!("../resources/spreadthewordlist.dict");

/// ingrid_core: Command-line crossword generation tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the grid file, as ASCII with # representing blocks and . representing empty squares
    grid_path: String,

    /// Path to a scored wordlist file [default: (embedded copy of Spread the Wordlist)]
    #[arg(long)]
    wordlist: Option<String>,

    /// Minimum allowable word score
    #[arg(long, default_value_t = 50)]
    min_score: u16,

    /// Maximum shared substring length between entries [default: none]
    #[arg(long)]
    max_shared_substring: Option<usize>,

    /// Print timing information along with the grid
    #[arg(short, long, default_value_t = false)]
    time: bool,

    /// Load a user word-list from PATH under the name LABEL (e.g., themeEntries:path/to/theme.dict)
    #[arg(long = "import-alt-list", value_name = "LABEL:PATH", num_args = 1.., action = clap::ArgAction::Append)]
    import_alt_lists: Option<Vec<String>>, 

    /// Constrain SLOTS (comma-sep, e.g., 1A,7D) to the theme list LABEL (e.g., themeEntries:1A,7D,12A)
    #[arg(long = "attach-alt-list", value_name = "LABEL:SLOTS", num_args = 1.., action = clap::ArgAction::Append)]
    attach_alt_lists: Option<Vec<String>>, 
}

// Custom Error struct for more context
struct AppError(String);

impl Debug for AppError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.0) 
    }
}

// Convert IngridWordListError to AppError
impl From<IngridWordListError> for AppError {
    fn from(err: IngridWordListError) -> Self {
        AppError(err.to_string())
    }
}
// Convert std::io::Error to AppError
impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError(err.to_string())
    }
}


fn main() -> Result<(), AppError> {
    let args = Args::parse();

    let raw_grid_content = fs::read_to_string(&args.grid_path)? 
        .lines()
        .map(|line| line.trim().to_lowercase().nfc().collect::<String>())
        .filter(|line| !line.is_empty()) 
        .collect::<Vec<_>>()
        .join("\n");

    let grid_lines: Vec<&str> = raw_grid_content.lines().collect();
    let height = grid_lines.len();

    if height == 0 {
        return Err(AppError("Grid must have at least one row".into()));
    }
    
    let width = grid_lines.first().map_or(0, |l| l.chars().count());
    if width == 0 {
        return Err(AppError("Grid must have at least one column".into()));
    }

    if grid_lines
        .iter()
        .any(|line| line.chars().count() != width) 
    {
        return Err(AppError("Rows in grid must all be the same length".into()));
    }

    let max_side = width.max(height);

    if !args
        .max_shared_substring
        .map_or(true, |mss| (3..=10).contains(&mss))
    {
        return Err(AppError(
            "If given, max shared substring must be between 3 and 10".into(),
        ));
    }

    let start_time = Instant::now(); 

    let main_wordlist_path = args.wordlist.map(PathBuf::from);
    let main_wordlist = WordList::new( 
        vec![if let Some(ref path) = main_wordlist_path {
            WordListSourceConfig::File {
                id: "main_default".into(), 
                enabled: true,
                path: path.clone().into(), 
            }
        } else {
            WordListSourceConfig::FileContents {
                id: "main_embedded".into(), 
                enabled: true,
                contents: STWL_RAW,
            }
        }],
        None, 
        Some(max_side),
        args.max_shared_substring,
    );

    let word_list_load_time = start_time.elapsed(); 

    let main_wl_id_check = if main_wordlist_path.is_some() { "main_default" } else { "main_embedded" };
    if let Some(errors) = main_wordlist.get_source_errors().get(main_wl_id_check) { 
        if !errors.is_empty() {
            let error_messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            return Err(AppError(format!(
                "Errors in main wordlist (source ID '{}'):\n- {}",
                main_wl_id_check, 
                error_messages.join("\n- ")
            )));
        }
    }
    
    if main_wordlist.word_id_by_string.is_empty() {
        return Err(AppError("Main word list is empty or failed to load.".into()));
    }

    // --- New functionality: Load Alternate Word Lists ---
    let mut imported_alt_lists: HashMap<String, WordList> = HashMap::new();
    if let Some(alt_list_specs) = args.import_alt_lists {
        for spec in alt_list_specs.iter() {
            let parts: Vec<&str> = spec.splitn(2, ':').collect();
            if parts.len() != 2 {
                 return Err(AppError(format!("import-alt-list needs LABEL:PATH, got: {spec}")));
            }
            let label = parts[0].to_string();
            let path_str = parts[1];
            
            let alt_wl = WordList::new(
                vec![WordListSourceConfig::File {
                    id: label.clone(), 
                    enabled: true,
                    path: path_str.into(),
                }],
                None, 
                Some(max_side), 
                None, 
            );

            if let Some(errors) = alt_wl.get_source_errors().get(&label) {
                 if !errors.is_empty() {
                    let error_messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                    return Err(AppError(format!(
                        "Errors in alt-list '{}' (path: {}):\n- {}",
                        label, path_str, error_messages.join("\n- ")
                    )));
                }
            }
            if alt_wl.word_id_by_string.is_empty() {
                 return Err(AppError(format!("Alternate word list '{label}' at '{path_str}' is empty or failed to load.",)));
            }
            imported_alt_lists.insert(label, alt_wl);
        }
    }
    
    // Generate initial grid_config (takes ownership of main_wordlist)
    let mut owned_grid_config =
        generate_grid_config_from_template_string(main_wordlist, &raw_grid_content, args.min_score);

    // --- New functionality: Attach Alternate Lists to Slots ---
    let mut slot_to_alt_list_label_map: HashMap<SlotId, String> = HashMap::new();
    if let Some(attach_specs) = args.attach_alt_lists {
        for spec in attach_specs.iter() {
            let parts: Vec<&str> = spec.splitn(2, ':').collect();
             if parts.len() != 2 {
                return Err(AppError(format!("attach-alt-list needs LABEL:SLOT1,SLOT2,..., got: {spec}")));
            }
            let label = parts[0];
            let slots_spec = parts[1];


            if !imported_alt_lists.contains_key(label) {
                return Err(AppError(format!(
                    "No --import-alt-list for label “{}” used in --attach-alt-list",
                    label
                )));
            }

            for s_token in slots_spec.split(',') {
                let s = s_token.trim();
                if s.is_empty() { continue; }

                let dir_char = s.chars().last().ok_or_else(|| AppError(format!("Invalid slot format: {s}")))?;
                let dir = if dir_char.eq_ignore_ascii_case(&'A') {
                    Direction::Across
                } else if dir_char.eq_ignore_ascii_case(&'D') {
                    Direction::Down
                } else {
                    return Err(AppError(format!(
                        "Slot direction must be 'A' or 'D', got: {dir_char} in {s}"
                    )));
                };

                let num_str = &s[..s.len() - 1];
                let num: usize = num_str.parse().map_err(|_| {
                    AppError(format!(
                        "Invalid slot number: {num_str} in {s}"
                    ))
                })?;

                let config_ref_for_slot_map = owned_grid_config.to_config_ref();
                let sid = config_ref_for_slot_map.slot_id_for_number_and_dir(num, dir).ok_or_else(|| {
                    AppError(format!("No such slot: {num}{}", if dir == Direction::Across {'A'} else {'D'}))
                })?;
                slot_to_alt_list_label_map.insert(sid, label.to_string());
            }
        }
    }

    // Override domains for themed slots
    if !slot_to_alt_list_label_map.is_empty() {
        let mut new_slot_options = owned_grid_config.slot_options.clone();

        for (&slot_id, list_label) in &slot_to_alt_list_label_map {
            let alt_list_ref = imported_alt_lists.get(list_label) // Immutable borrow to read words
                .ok_or_else(|| AppError(format!("Internal error: Alt list label '{list_label}' not found after check.")))?;

            let slot_cfg = &owned_grid_config.slot_configs[slot_id]; // From owned_grid_config
            let entry_fill_vec = slot_cfg.fill(&owned_grid_config.fill, owned_grid_config.width);
            
            let (num_debug, dir_debug_char) = owned_grid_config.slot_map.iter().find_map(|(&(n,d), &s_id)| {
                if s_id == slot_id { Some((n, if d == Direction::Across {'A'} else {'D'})) } else { None }
            }).unwrap_or((slot_cfg.number, '?')); // Fallback to slot_cfg.number if not in map (should be)


            eprintln!(
                "DEBUG: Constraining slot {}{}{} (SlotId: {}, Length: {}) with alt-list '{}'",
                num_debug, dir_debug_char, 
                if dir_debug_char == '?' { "(dir unknown)" } else { "" }, 
                slot_id, slot_cfg.length, list_label
            );

            let mut candidate_words_from_alt_list: Vec<IngridWord> = Vec::new();
            if slot_cfg.length < alt_list_ref.words.len() {
                for word_obj_in_alt in &alt_list_ref.words[slot_cfg.length] {
                    if word_obj_in_alt.hidden { continue; }
                    let matches_fill = entry_fill_vec.iter().enumerate().all(|(idx, fill_glyph_opt)| {
                        fill_glyph_opt.map_or(true, |fill_glyph| {
                            word_obj_in_alt.glyphs.get(idx).map_or(false, |&word_glyph| word_glyph == fill_glyph)
                        })
                    });
                    if matches_fill {
                        candidate_words_from_alt_list.push(word_obj_in_alt.clone());
                    }
                }
            }
             eprintln!(
                "DEBUG: Slot {}{}: Found {} candidate words from alt-list '{}' (matching length & fill). First few: {:?}",
                num_debug, dir_debug_char, candidate_words_from_alt_list.len(), list_label,
                candidate_words_from_alt_list.iter().take(3).map(|w| &w.normalized_string).collect::<Vec<_>>()
            );


            let mut main_list_word_ids_for_slot: Vec<WordId> = Vec::new();
            for word_to_transfer in candidate_words_from_alt_list {
                // Ensure the word exists in the main WordList and get its ID.
                // This call mutates owned_grid_config.word_list if the word or its glyphs are new.
                let (_len, main_wl_word_id) = owned_grid_config.word_list.get_word_id_or_add_hidden(&word_to_transfer.normalized_string);
                
                // Ensure this word is usable (not hidden, has a score) in the main list.
                if main_wl_word_id < owned_grid_config.word_list.words[slot_cfg.length].len() {
                     let word_in_main_list = &mut owned_grid_config.word_list.words[slot_cfg.length][main_wl_word_id];
                     word_in_main_list.hidden = false; // Unhide it
                     // Use the score from the theme list, as it's explicitly provided for this theme.
                     word_in_main_list.score = word_to_transfer.score; 
                     // Keep canonical string from theme list if desired (might differ from main list's version)
                     word_in_main_list.canonical_string = word_to_transfer.canonical_string.clone(); 
                } else {
                    eprintln!("WARN: WordId {} for word '{}' out of bounds for main wordlist length {} after get_word_id_or_add_hidden. Skipping.", main_wl_word_id, word_to_transfer.normalized_string, slot_cfg.length);
                    continue;
                }
                main_list_word_ids_for_slot.push(main_wl_word_id);
            }
            
            new_slot_options[slot_id] = main_list_word_ids_for_slot;
            
            // Corrected eprintln! macro call
            eprintln!(
                "DEBUG: Slot {}{}: Finalized to {} options in main wordlist context (from alt-list '{}'). WordIDs (main list): {:?}",
                num_debug, 
                dir_debug_char, 
                new_slot_options[slot_id].len(), 
                list_label, // Argument for the {}
                new_slot_options[slot_id].iter().take(5).collect::<Vec<_>>() // Argument for the {:?}
            );


            if new_slot_options[slot_id].is_empty() {
                 return Err(AppError(format!(
                    "Slot {}{} (ID {}) constrained by alt-list '{}' has no valid words from that list that could be used (Length: {}). Check for conflicts or ensure words exist in main list if pre-seeded.",
                    num_debug, dir_debug_char, slot_id, list_label, slot_cfg.length
                )));
            }
        }
        owned_grid_config.slot_options = new_slot_options;
        
        ingrid_core::grid_config::sort_slot_options(
            &owned_grid_config.word_list, 
            &owned_grid_config.slot_configs,
            &mut owned_grid_config.slot_options
        );
    }
    
    let final_config_ref = owned_grid_config.to_config_ref();
    let result = find_fill(&final_config_ref, None, None)
        .map_err(|e| AppError(format!("Unfillable grid: {:?}", e)))?;

    let fill_time = start_time.elapsed() - word_list_load_time;

    println!(
        "{}",
        render_grid(&final_config_ref, &result.choices)
    );

    if args.time {
        eprintln!("{word_list_load_time:?} loading word list(s), {fill_time:?} finding fill");
    }

    Ok(())
}
