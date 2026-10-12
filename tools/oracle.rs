use poker_engine::table::eval::eval_nlh_river;
use poker_engine::Card;
use std::io::{BufWriter, Write};

// These physical hands match checks/s2_3.py, including an exact pair tie.
const ANCHORS: [[u8; 7]; 16] = [
    [12,23,34,45,4,15,26], [12,25,28,42,17,33,47],
    [12,25,11,24,1,14,2], [7,20,33,12,23,48,0],
    [12,24,36,48,8,0,14], [10,8,6,4,2,0,25],
    [1,14,27,0,13,34,49], [0,13,26,39,1,14,27],
    [12,11,10,9,8,20,21], [0,13,26,39,12,25,11],
    [11,24,37,12,25,38,0], [12,25,11,24,10,23,0],
    [12,0,14,28,42,20,34], [12,0,1,2,3,20,21],
    [12,25,26,40,17,33,47], [0,1,2,3,7,25,34],
];

// Explicit algorithm/version makes sampling independent of rand crate versions.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, bound: u64) -> usize {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next();
            if x >= threshold { return (x % bound) as usize; }
        }
    }
    fn hand(&mut self) -> [u8; 7] {
        let mut deck = [0u8; 52];
        for (i, card) in deck.iter_mut().enumerate() { *card = i as u8; }
        for i in 0..7 {
            let j = i + self.below((52 - i) as u64);
            deck.swap(i, j);
        }
        deck[..7].try_into().unwrap()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 { return Err("usage: poker_huy_oracle OUTPUT COUNT SEED".into()); }
    let count: u64 = args[2].parse()?;
    let seed: u64 = args[3].parse()?;
    if count < ANCHORS.len() as u64 { return Err("COUNT must include all 16 anchors".into()); }
    let mut out = BufWriter::new(std::fs::File::create(&args[1])?);
    out.write_all(b"PHUYOR01")?;
    out.write_all(&count.to_le_bytes())?;
    out.write_all(&seed.to_le_bytes())?;
    let mut rng = SplitMix64(seed);
    for i in 0..count {
        let hand = if i < ANCHORS.len() as u64 { ANCHORS[i as usize] } else { rng.hand() };
        let rust = hand.map(|id| Card(((id % 13) << 2) | (3 - id / 13)));
        let rank = eval_nlh_river([rust[0], rust[1]], [rust[2],rust[3],rust[4],rust[5],rust[6]]).0;
        if !(1..=7462).contains(&rank) { return Err(format!("invalid oracle rank {rank} at hand {i}").into()); }
        out.write_all(&hand)?;
        out.write_all(&rank.to_le_bytes())?;
    }
    out.flush()?;
    println!("Rust oracle: {count} hands, seed {seed}, 16 fixed anchors");
    Ok(())
}
