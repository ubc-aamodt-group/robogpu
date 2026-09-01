#ifndef RT_FUNC_UNITS_INCLUDED
#define RT_FUNC_UNITS_INCLUDED

#include "../abstract_hardware_model.h"

typedef struct warp_thread_id {
    unsigned warp_uid;
    unsigned tid;
    bool predicate;
} warp_thread_id;

typedef struct op_unit_stats {
    unsigned total_ops = 0;
    unsigned predicated_off_ops = 0;
    unsigned long long total_active_cycles = 0;
    unsigned long long pipeline_stalled_cycles = 0;
    unsigned long long input_stalled_cycles = 0;
    unsigned long long total_op_cycles = 0;
} op_unit_stats;

typedef struct op_unit_aerialvision {
    unsigned current_op_count = 0;
    unsigned input_queue_size = 0;
} op_unit_aerialvision;

class func_unit_stats {
  public:
    func_unit_stats(unsigned n_units) {
        m_n_units = n_units;
        total_intersections = (unsigned *)calloc(n_units, sizeof(unsigned));
        output_stalled_cycles = (unsigned *)calloc(n_units, sizeof(unsigned));
        output_queue_size = (unsigned *)calloc(n_units, sizeof(unsigned));

        intersection_cycles = (unsigned long long *)calloc(static_cast<int>(TransactionType::UNDEFINED), sizeof(unsigned long long));
        intersection_count = (unsigned *)calloc(static_cast<int>(TransactionType::UNDEFINED), sizeof(unsigned));

        detailed_intersection_cycles = (unsigned long long **)calloc(static_cast<int>(TransactionType::UNDEFINED), sizeof(unsigned long long *));

        per_op_stats = (op_unit_stats **)calloc(n_units, sizeof(op_unit_stats *));
        interconnect_stats = (op_unit_stats *)calloc(n_units, sizeof(op_unit_stats));

        failed_arbitrations = 0;
    }
    ~func_unit_stats();

    void print(FILE *fout);
    op_unit_stats* init_per_op(unsigned unit_id, unsigned n_op_units) {
        m_n_op_units = n_op_units;
        per_op_stats[unit_id] = (op_unit_stats *)calloc(n_op_units, sizeof(op_unit_stats));

        for (int i = 0; i < static_cast<int>(TransactionType::UNDEFINED); i++) {
            detailed_intersection_cycles[i] = (unsigned long long *)calloc(n_op_units+1, sizeof(unsigned long long));
        }
        return per_op_stats[unit_id];
    }

    unsigned m_n_units;
    unsigned m_n_op_units;
    unsigned* total_intersections;

    unsigned failed_arbitrations;
    unsigned* output_stalled_cycles;
    unsigned* output_queue_size;

    // Per function stats
    unsigned long long* intersection_cycles;
    unsigned long long** detailed_intersection_cycles;
    unsigned* intersection_count; 

    // Per op unit stats
    op_unit_stats** per_op_stats;
    op_unit_stats* interconnect_stats;
};

class rt_op_unit {
    public:
        rt_op_unit(unsigned sid, unsigned unit_id, unsigned latency, unsigned init_cycles, RTFuncInsnType op, std::string unit_name, unsigned n_units, bool is_cluster, op_unit_stats* stats);
        ~rt_op_unit();

        void cycle();

        void add_input(warp_thread_id thread) { m_input_queue.push(thread); }
        bool is_ready() { return m_waiting_cycles == 0; }
        bool has_output() { return m_output_queue.size() > 0; }
        warp_thread_id get_output() { 
            warp_thread_id next_thread = m_output_queue.front();
            m_output_queue.pop();
            return next_thread;
        }
        void reset_output_queue(std::queue<warp_thread_id> new_queue) { m_output_queue = new_queue; }
        std::string get_unit_name() { return m_unit_name; }
        RTFuncInsnType get_op() { return m_op; }
        op_unit_aerialvision get_aerialvision_stats() { return m_aerialvision_stats; }

        unsigned get_current_count() { return m_current_execution.size() + m_input_queue.size() + m_output_queue.size(); }

    private:
        unsigned m_latency;
        unsigned m_init_cycles;
        RTFuncInsnType m_op;
        unsigned* m_waiting_cycles;
        unsigned m_n_units;
        op_unit_stats* m_stats;
        op_unit_aerialvision m_aerialvision_stats;

        unsigned m_sid;
        unsigned m_unit_id;
        std::string m_unit_name;
        bool m_is_cluster;

        std::queue<warp_thread_id> m_input_queue;
        std::queue<warp_thread_id> m_output_queue;

        std::vector<std::pair<warp_thread_id, unsigned> > m_current_execution;
};


class rt_func_unit {
    public:
        rt_func_unit(unsigned sid, rt_func_unit_config config, func_unit_stats* stats, unsigned unit_id);
        ~rt_func_unit();

        void cycle(std::map<unsigned, warp_inst_t>& warps, std::deque<warp_thread_id>& awaiting_threads, std::deque<std::pair<unsigned, new_addr_type> > &store_queue);
        unsigned count_in_progress() { return m_active_threads; }
        void get_aerialvision_stats(op_unit_aerialvision* stats);
        void reset_op_unit_arb();

        unsigned count_op(RTFuncInsnType op);

    private:
        rt_func_unit_config m_config;
        func_unit_stats* m_stats;
        op_unit_stats* m_op_stats;
        std::vector<rt_op_unit*> m_op_units;
        rt_op_unit* m_interconnect_unit;
        unsigned m_active_threads;

        // std::bitset<static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE)> m_op_unit_arb;
        std::map<unsigned, unsigned> m_op_unit_arb;
        unsigned m_last_op;

        std::map<unsigned, unsigned long long> latency_tracker;
        std::map<unsigned, unsigned long long> detailed_latency_tracker;

        unsigned m_sid;
        unsigned m_unit_id;
};


#endif
