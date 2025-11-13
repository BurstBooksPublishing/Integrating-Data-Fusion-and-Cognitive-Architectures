#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 

static void timespec_add_ms(struct timespec *t, long ms){
  t->tv_sec += ms / 1000;
  t->tv_nsec += (ms % 1000) * 1000000L;
  if (t->tv_nsec >= 1000000000L){ t->tv_sec++; t->tv_nsec -= 1000000000L; }
}

static int set_realtime(int prio, int cpu){
  struct sched_param p = {.sched_priority = prio};
  if (sched_setscheduler(0, SCHED_FIFO, &p)) return -1;
  cpu_set_t cp;
  CPU_ZERO(&cp); CPU_SET(cpu, &cp);
  return sched_setaffinity(0, sizeof(cp), &cp);
}

void *periodic_worker(void *arg){
  const int period_ms = *((int*)arg);
  struct timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  timespec_add_ms(&next, period_ms);

  for (int iter=0; iter<1000; ++iter){
    // sleep until absolute next time (avoids accumulation error)
    int r = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    if (r && r!=EINTR){ perror("clock_nanosleep"); break; }

    struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
    // jitter = actual - expected in microseconds
    int64_t jitter_us = (now.tv_sec - next.tv_sec) * 1000000LL
                       + (now.tv_nsec - next.tv_nsec) / 1000LL;
    // simulate work bounded by WCET (here a busy loop placeholder)
    // In production, replace with real compute and measured WCET guards.
    // ... perform fusion/cognition operator ...
    printf("iter %d jitter(us) %lld\n", iter, (long long)jitter_us);

    timespec_add_ms(&next, period_ms);
  }
  return NULL;
}

int main(){
  pthread_t thr; int period = 10; // 10 ms period
  if (set_realtime(80, 0)){ perror("realtime"); /* continue but expect non-deterministic */ }
  if (pthread_create(&thr, NULL, periodic_worker, &period)){ perror("pthread"); return 1; }
  pthread_join(thr, NULL);
  return 0;
}