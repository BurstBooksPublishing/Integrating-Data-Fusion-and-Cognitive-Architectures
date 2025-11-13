#include 
#include 

// read clocks; convert to seconds
double to_sec(const timespec &ts){ return ts.tv_sec + ts.tv_nsec*1e-9; }

int main(){
  timespec tm, tr;
  clock_gettime(CLOCK_MONOTONIC_RAW, &tm); // monotonic for ordering
  clock_gettime(CLOCK_REALTIME, &tr);      // wall-time (may be stepped)
  // In production, obtain offset from PTP daemon (simulated here).
  double ptp_offset_seconds = to_sec(tr) - to_sec(tm); // small drift/offset
  double monotonic_s = to_sec(tm);
  double wall_estimate = monotonic_s + ptp_offset_seconds; // disciplined wall time
  // attach both to message for fusion and auditing
  std::cout << "monotonic:" << monotonic_s << " wall_est:" << wall_estimate << "\n";
  return 0;
}