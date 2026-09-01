#include "rt_func_units.h"
#include "../../libcuda/gpgpu_context.h"


std::string RTFuncInsnTypeNames[] = {
    "RT_DECODE",
    "RT_VEC_SUB",
    "RT_RCP",
    "RT_MUL",
    "RT_MAXMIN",
    "RT_MINMAX",
    "RT_VEC_CMP",
    "RT_VEC_OR",
    "RT_CROSS",
    "RT_DOT",
    "RT_SQRT",
    "RT_RAY_BOX_FUNC_OP",
    "RT_RAY_TRI_FUNC_OP",
    "RT_RAY_XFORM_FUNC_OP",
    "RT_RETURN",
    "RT_OP_CLUSTER_A",
    "RT_OP_CLUSTER_B",
    "RT_OP_CLUSTER_C"
};

rt_op_unit::rt_op_unit(unsigned sid, unsigned unit_id, unsigned latency, unsigned init_cycles, RTFuncInsnType op, std::string unit_name, unsigned n_units, bool is_cluster, op_unit_stats* stats) {
    m_sid = sid;
    m_unit_id = unit_id;
    m_unit_name = unit_name;
    m_n_units = n_units;
    m_latency = latency;
    m_init_cycles = init_cycles;
    m_op = op;
    m_is_cluster = is_cluster;
    m_waiting_cycles = (unsigned *)calloc(n_units, sizeof(unsigned));
    m_stats = stats;
    m_aerialvision_stats.current_op_count = 0;
    m_aerialvision_stats.input_queue_size = 0;

    if (m_sid == 0) {
        printf("[%d:%d] Created %d %s unit with latency %d and init_cycles %d\n", m_sid, m_unit_id, m_n_units, m_unit_name.c_str(), m_latency, m_init_cycles);
    }
}

void rt_op_unit::cycle() {
    for (unsigned i=0; i<m_n_units; i++) {
        if (m_waiting_cycles[i] > 0) {
            m_waiting_cycles[i]--;
            if (m_input_queue.size() > 0) {
                m_stats->pipeline_stalled_cycles++;
            }
        }
        else {
            if (m_input_queue.size() > 0) {
                warp_thread_id next_thread = m_input_queue.front();

                unsigned latency = m_latency;
                if (next_thread.predicate == false) {
                    latency = 1; // Predicated off threads take 1 cycle
                    m_stats->predicated_off_ops++;
                    // printf("Predicated off [%d:%d] in %s.\n", next_thread.warp_uid, next_thread.tid, m_unit_name.c_str());
                }

                // TODO: Clusters have variable latency operations; handle separately
                if (m_is_cluster) {
                    m_current_execution.push_back(std::pair<warp_thread_id, unsigned>(next_thread, latency));
                    m_input_queue.pop();
                    // printf("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_unit_name.c_str());
                    m_waiting_cycles[i] = m_init_cycles - 1; // -1 for current cycle
                }
                else {
                    m_current_execution.push_back(std::pair<warp_thread_id, unsigned>(next_thread, latency));
                    m_input_queue.pop();
                    // printf("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_unit_name.c_str());
                    m_waiting_cycles[i] = m_init_cycles - 1; // -1 for current cycle
                }
            }
        }
    }
    if (m_input_queue.size() > 0) {
        m_stats->input_stalled_cycles++;
    }
    if (m_current_execution.size() > 0) {
        m_stats->total_active_cycles++;
    }

    // Cycle pipeline
    for (auto it=m_current_execution.begin(); it!=m_current_execution.end();) {
        if (it->second > 0) {
            it->second--;
            it++;
        }
        else {
            m_output_queue.push(it->first);
            it = m_current_execution.erase(it);
            // printf("[%d:%d] finished in %s.\n", it->first.warp_uid, it->first.tid, m_unit_name.c_str());
            m_stats->total_ops++;
        }
    }

    // Track this cycle for everything in the pipeline
    m_stats->total_op_cycles += m_current_execution.size() + m_input_queue.size();

    // Update AerialVision stats
    m_aerialvision_stats.current_op_count = m_current_execution.size();
    m_aerialvision_stats.input_queue_size = m_input_queue.size();
}

rt_func_unit::rt_func_unit(unsigned sid, rt_func_unit_config config, func_unit_stats* stats, unsigned unit_id) {
    m_sid = sid;
    m_unit_id = unit_id;
    m_config = config;
    m_active_threads = 0;
    m_stats = stats;

    m_op_stats = m_stats->init_per_op(m_unit_id, static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE));
    m_last_op = 0;

    // Iterate through each op type
    if (m_sid == 0) {
        printf("Creating RT func unit %d with %d op units.\n", m_unit_id, static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE));
    }
    unsigned op_unit_id = 0;
    for (unsigned i=0; i<static_cast<unsigned>(RTFuncInsnType::RT_OP_CLUSTER_A); i++) {
        unsigned n_units = m_config.n_units[i];
        unsigned latency = m_config.latency[i];
        unsigned init_cycles = m_config.initiation_cycles[i];
        bool is_cluster = false;

        rt_op_unit* new_unit = new rt_op_unit(m_sid, op_unit_id, latency, init_cycles, static_cast<RTFuncInsnType>(i), RTFuncInsnTypeNames[i], n_units, is_cluster, &m_op_stats[i]);
        m_op_units.push_back(new_unit);
        op_unit_id++;

        m_op_unit_arb[i] = m_config.n_units[i];
    }

    if (m_sid == 0) {
        printf("Creating RT cluster units...\n");
    }
    for (unsigned i=static_cast<unsigned>(RTFuncInsnType::RT_OP_CLUSTER_A); i<static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE); i++) {
        unsigned n_units = m_config.n_units[i];
        unsigned latency = m_config.latency[i];
        unsigned init_cycles = m_config.initiation_cycles[i];
        bool is_cluster = true;

        rt_op_unit* new_unit = new rt_op_unit(m_sid, op_unit_id, latency, init_cycles, static_cast<RTFuncInsnType>(i), RTFuncInsnTypeNames[i], n_units, is_cluster, &m_op_stats[i]);
        m_op_units.push_back(new_unit);
        op_unit_id++;

        m_op_unit_arb[i] = m_config.n_units[i];
    }

    m_interconnect_unit = new rt_op_unit(m_sid, op_unit_id, m_config.interconnect_latency, m_config.interconnect_init_cycles, RTFuncInsnType::RT_MAX_INSN_TYPE, "interconnect", m_op_units.size(), false, &m_stats->interconnect_stats[m_unit_id]);
}

void rt_func_unit::reset_op_unit_arb() {
    for (unsigned i=0; i<static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE); i++) {
        m_op_unit_arb[i] = m_config.n_units[i];
    }
}

void rt_func_unit::cycle(std::map<unsigned, warp_inst_t>& warps, std::deque<warp_thread_id>& awaiting_threads, std::deque<std::pair<unsigned, new_addr_type> > &store_queue) {

    for (unsigned i=0; i<m_config.func_bw; i++) {
        if (awaiting_threads.size() > 0) {
            warp_thread_id next_thread = awaiting_threads.front();

            // Make an uid (tid is 32 max, which is 5 bits)
            unsigned next_thread_uid = next_thread.warp_uid << 5 | next_thread.tid;
            unsigned long long current_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
            latency_tracker[next_thread_uid] = current_cycle;
            detailed_latency_tracker[next_thread_uid] = current_cycle;

            m_active_threads++;
            awaiting_threads.pop_front();

            RTFuncInsnType next_op = warps[next_thread.warp_uid].get_next_op(next_thread.tid);
            m_op_units[static_cast<unsigned>(next_op)]->add_input(next_thread);

            RT_DPRINTF("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_op_units[static_cast<unsigned>(next_op)]->get_unit_name().c_str());
        }
    }

    // Use round robin for now
    for (unsigned i=0; i<m_op_units.size(); i++) {
        unsigned op_unit_id = (m_last_op + i) % m_op_units.size();
        m_op_units[op_unit_id]->cycle();

        // Process m_n_units outputs
        unsigned n_units = m_config.n_units[op_unit_id];
        for (unsigned j=0; j<n_units; j++) {
            if (m_op_units[op_unit_id]->has_output()) {
                warp_thread_id next_thread = m_op_units[op_unit_id]->get_output();
                bool done = warps[next_thread.warp_uid].dec_thread_latency(next_thread.tid);

                unsigned next_thread_uid = next_thread.warp_uid << 5 | next_thread.tid;
                unsigned long long current_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
                TransactionType intersection_type = warps[next_thread.warp_uid].get_current_transaction_type(next_thread.tid);

                if (done) {
                    // Get stores
                    std::deque<std::pair<unsigned, new_addr_type> > st_queue = warps[next_thread.warp_uid].check_stores(next_thread.tid);
                    std::copy(st_queue.begin(), st_queue.end(), std::back_inserter(store_queue));

                    // Remove from func_unit
                    m_active_threads--;
                    m_stats->total_intersections[m_unit_id]++;

                    // Remove from latency tracker
                    unsigned long long latency = current_cycle - latency_tracker[next_thread_uid];
                    latency_tracker.erase(next_thread_uid);

                    m_stats->intersection_cycles[static_cast<unsigned>(intersection_type)] += latency;
                    m_stats->intersection_count[static_cast<unsigned>(intersection_type)]++;

                    unsigned long long detailed_latency = current_cycle - detailed_latency_tracker[next_thread_uid];
                    detailed_latency_tracker.erase(next_thread_uid);

                    m_stats->detailed_intersection_cycles[static_cast<unsigned>(intersection_type)][op_unit_id] += detailed_latency;

                    // printf("[%d:%d] finished\n", next_thread.warp_uid, next_thread.tid);
                }
                else {
                    // Add back to latency tracker
                    unsigned long long latency = current_cycle - detailed_latency_tracker[next_thread_uid];
                    m_stats->detailed_intersection_cycles[static_cast<unsigned>(intersection_type)][op_unit_id] += latency;
                    detailed_latency_tracker[next_thread_uid] = current_cycle;

                    RTFuncInsnType next_op = warps[next_thread.warp_uid].get_next_op(next_thread.tid);
                    
                    // Borrow RT_RETURN as predicate off marker
                    if (next_op == RTFuncInsnType::RT_RETURN) {
                        next_thread.predicate = false;
                        // printf("Predicating off [%d:%d]\n", next_thread.warp_uid, next_thread.tid);
                        bool config_pred = GPGPU_Context()->func_sim->g_rt_predicate_return;
                        if (config_pred) {
                            // Ignore RT_RETURN (assume that RT_RETURN is never the last op)
                            warps[next_thread.warp_uid].dec_thread_latency(next_thread.tid);
                            next_op = warps[next_thread.warp_uid].get_next_op(next_thread.tid);
                        }
                    }

                    int func_type = GPGPU_Context()->func_sim->g_rt_func_type;
                    // Check if interconnect is needed (skip if next op matches current op)
                    if (static_cast<unsigned>(next_op) == op_unit_id) {
                        m_op_units[op_unit_id]->add_input(next_thread);
                    }
                    // Special case for staged collision core / any g_rt_func_type == 0 fixed function core should not require interconnect
                    else if (func_type == 0) {
                        m_op_units[static_cast<unsigned>(next_op)]->add_input(next_thread);
                    }
                    // Otherwise, send to interconnect
                    else {
                        m_interconnect_unit->add_input(next_thread);
                    }
                }
            }
        }
    }
    m_last_op = (m_last_op + 1) % m_op_units.size();

    // Cycle each op unit
    reset_op_unit_arb();
    std::queue<warp_thread_id> interconnect_queue;
    while (m_interconnect_unit->has_output()) {
        warp_thread_id next_thread = m_interconnect_unit->get_output();

        // Check arbitration
        RTFuncInsnType next_op = warps[next_thread.warp_uid].get_next_op(next_thread.tid);
        if (m_op_unit_arb[static_cast<unsigned>(next_op)] < 1) {
            // Failed arbitration
            m_stats->failed_arbitrations++;
            interconnect_queue.push(next_thread);
        }
        else {
            m_op_unit_arb[static_cast<unsigned>(next_op)]--;
            m_op_units[static_cast<unsigned>(next_op)]->add_input(next_thread);
            // printf("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_op_units[static_cast<unsigned>(next_op)]->get_unit_name().c_str());

            unsigned next_thread_uid = next_thread.warp_uid << 5 | next_thread.tid;
            unsigned long long current_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
            unsigned long long latency = current_cycle - detailed_latency_tracker[next_thread_uid];
            TransactionType intersection_type = warps[next_thread.warp_uid].get_current_transaction_type(next_thread.tid);

            m_stats->detailed_intersection_cycles[static_cast<unsigned>(intersection_type)][m_op_units.size()] += latency;
            detailed_latency_tracker[next_thread_uid] = current_cycle;
        }
    }

    if (interconnect_queue.size() > 0) {
        m_stats->output_stalled_cycles[m_unit_id]++;
        m_stats->output_queue_size[m_unit_id] += interconnect_queue.size();
        m_interconnect_unit->reset_output_queue(interconnect_queue);
    }

    // Cycle interconnect unit (for tracking latency)
    m_interconnect_unit->cycle();
}

unsigned rt_func_unit::count_op(RTFuncInsnType op) {
    unsigned count = 0;
    for (unsigned i=0; i<m_op_units.size(); i++) {
        if (m_op_units[i]->get_op() == op) {
            count += m_op_units[i]->get_current_count();
        }
    }
    return count;
}

void rt_func_unit::get_aerialvision_stats(op_unit_aerialvision* aerialvision_stats) {
    op_unit_aerialvision interconnect_stats = m_interconnect_unit->get_aerialvision_stats();
    op_unit_aerialvision total_stats = interconnect_stats;

    for (unsigned i=0; i<m_op_units.size(); i++) {
        aerialvision_stats[i] = m_op_units[i]->get_aerialvision_stats();
        total_stats.current_op_count += aerialvision_stats[i].current_op_count;
        total_stats.input_queue_size += aerialvision_stats[i].input_queue_size;
    }

    aerialvision_stats[m_op_units.size()] = interconnect_stats;
    aerialvision_stats[m_op_units.size()+1] = total_stats;
}

void func_unit_stats::print(FILE *fout) {
    unsigned intersections = 0;
    unsigned interconnect = 0;
    unsigned interconnect_stalled_cycles = 0;
    fprintf(fout, "Per func unit intersections: ");
    for (unsigned i=0; i<m_n_units; i++) {
        fprintf(fout, "%d ", total_intersections[i]);
        intersections += total_intersections[i];
        interconnect += output_queue_size[i];
        interconnect_stalled_cycles += output_stalled_cycles[i];
    }
    fprintf(fout, "\n");
    fprintf(fout, "total_intersections = %d\n", intersections);
    fprintf(fout, "failed_arbitrations = %d\n", failed_arbitrations);
    fprintf(fout, "avg_stalled_output_queue_size = %f\n", (float)interconnect / interconnect_stalled_cycles);

    std::vector<unsigned> transaction_types;
    for (unsigned t=0; t<static_cast<unsigned>(TransactionType::UNDEFINED); t++) {
        float avg_latency = (float)intersection_cycles[t] / (float)intersection_count[t];
        if (!std::isnan(avg_latency)) {
            fprintf(fout, "avg_latency[%d] = %f\n", t, avg_latency);
            transaction_types.push_back(t);
        }
    }

    for (auto t : transaction_types) {
        fprintf(fout, "detailed_avg_latency[%d] = ", t);
        for (unsigned i=0; i<m_n_op_units+1; i++) {
            float avg_latency = (float)detailed_intersection_cycles[t][i] / (float)intersection_count[t];
            if (!std::isnan(avg_latency)) {
                fprintf(fout, "%f ", avg_latency);
            }
            else {
                fprintf(fout, "0 ");
            }
        }
        fprintf(fout, "\n");
    }

    fprintf(fout, "Interconnect stats:\n");
    fprintf(fout, "total_active_cycles = ");
    for (unsigned i=0; i<m_n_units; i++) {
        fprintf(fout, "%llu ", interconnect_stats[i].total_active_cycles);
    }
    fprintf(fout, "\n");
    fprintf(fout, "pipeline_stalled_cycles = ");
    for (unsigned i=0; i<m_n_units; i++) {
        fprintf(fout, "%llu ", interconnect_stats[i].pipeline_stalled_cycles);
    }
    fprintf(fout, "\n");
    fprintf(fout, "input_stalled_cycles = ");
    for (unsigned i=0; i<m_n_units; i++) {
        fprintf(fout, "%llu ", interconnect_stats[i].input_stalled_cycles);
    }
    fprintf(fout, "\n");
    fprintf(fout, "avg_op_latency = ");
    for (unsigned i=0; i<m_n_units; i++) {
        fprintf(fout, "%f ", interconnect_stats[i].total_op_cycles / (float)interconnect_stats[i].total_ops);
    }
    fprintf(fout, "\n");

    for (unsigned i=0; i<m_n_op_units; i++) {
        fprintf(fout, "total_active_cycles[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%llu ", per_op_stats[j][i].total_active_cycles);
        }
        fprintf(fout, "\n");
        fprintf(fout, "pipeline_stalled_cycles[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%llu ", per_op_stats[j][i].pipeline_stalled_cycles);
        }
        fprintf(fout, "\n");
        fprintf(fout, "input_stalled_cycles[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%llu ", per_op_stats[j][i].input_stalled_cycles);
        }
        fprintf(fout, "\n");
        fprintf(fout, "avg_op_latency[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%f ", per_op_stats[j][i].total_op_cycles / (float)per_op_stats[j][i].total_ops);
        }
        fprintf(fout, "\n");
        fprintf(fout, "avg_predicated_ops[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%f ", per_op_stats[j][i].predicated_off_ops / (float)per_op_stats[j][i].total_ops);
        }
        fprintf(fout, "\n");
    }

}
