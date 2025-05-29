`timescale 1 ns / 1 ps

// ============================================================================
// NPU AXI4-Lite Peripheral - real_npu_ip_v1_0_S00_AXI
//  - Implements a 2x2 matrix multiply with optional ReLU activation
//  - AXI4-Lite slave interface for control and data transfer
//  - Results exposed in output registers, control via FSM
// ============================================================================
module real_npu_ip_v1_0_S00_AXI #
(
    parameter integer C_S_AXI_DATA_WIDTH = 32, // AXI data bus width
    parameter integer C_S_AXI_ADDR_WIDTH = 6   // AXI address bus width
)
(
    // --- AXI4-Lite Slave Interface Signals ---
    input  wire S_AXI_ACLK,                      // AXI clock
    input  wire S_AXI_ARESETN,                   // AXI active-low reset

    // Write address channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  wire [2:0] S_AXI_AWPROT,
    input  wire S_AXI_AWVALID,
    output wire S_AXI_AWREADY,

    // Write data channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  wire S_AXI_WVALID,
    output wire S_AXI_WREADY,

    // Write response channel
    output wire [1:0] S_AXI_BRESP,
    output wire S_AXI_BVALID,
    input  wire S_AXI_BREADY,

    // Read address channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  wire [2:0] S_AXI_ARPROT,
    input  wire S_AXI_ARVALID,
    output wire S_AXI_ARREADY,

    // Read data channel
    output wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output wire [1:0] S_AXI_RRESP,
    output wire S_AXI_RVALID,
    input  wire S_AXI_RREADY
);

    // =========================================================================
    // AXI Register Map & Internal Registers
    // =========================================================================
    // slv_reg0: Control register (start, clear_done, relu_en)
    // slv_reg1: Status register (busy, done)
    // slv_reg2-5: Input matrices (A and B, 2 elements packed per register)
    // slv_reg6-9: Output results (each is one product, 4 in total)
    // slv_regA: Debug register
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg0, slv_reg1, slv_reg2, slv_reg3, slv_reg4, slv_reg5;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg6, slv_reg7, slv_reg8, slv_reg9; // Output regs

    // Write enable signal for slave registers
    wire slv_reg_wren;
    reg [C_S_AXI_DATA_WIDTH-1:0] reg_data_out; // Output mux for reads
    integer byte_index; // Used in byte write operations

    // =========================================================================
    // AXI Addressing and Control
    // =========================================================================
    // All addresses are word aligned (4 bytes per location)
    localparam ADDR_LSB = 2;
    localparam OPT_MEM_ADDR_BITS = 3;   // 8 registers (minimum)
    localparam NUM_REG_BITS = 4;        // 16 registers (max supported)

    // AXI state variables and registers
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg axi_awready, axi_wready, axi_bvalid;
    reg [1:0] axi_bresp;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg axi_arready, axi_rvalid;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0] axi_rresp;
    reg aw_en;
    reg [31:0] debug_reg; // Debug/visibility register

    // =========================================================================
    // AXI Output Assignments
    // =========================================================================
    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    // =========================================================================
    // AXI Write Address Channel FSM
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            aw_en <= 1'b1; // Allow next write
        end else begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en) begin
                axi_awready <= 1'b1; // Accept write address
                aw_en <= 1'b0;
            end else if (S_AXI_BREADY && axi_bvalid) begin
                aw_en <= 1'b1; // Re-enable after write response
                axi_awready <= 1'b0;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end

    // Latch AXI write address
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            axi_awaddr <= 0;
        else if (axi_awready && S_AXI_AWVALID)
            axi_awaddr <= S_AXI_AWADDR;
    end

    // Write data channel FSM
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            axi_wready <= 1'b0;
        else if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en)
            axi_wready <= 1'b1;
        else
            axi_wready <= 1'b0;
    end

    // Slave register write enable
    assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    // =========================================================================
    // Write to Slave Registers (Control, Input)
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            slv_reg0 <= 0;
            slv_reg2 <= 0;
            slv_reg3 <= 0;
            slv_reg4 <= 0;
            slv_reg5 <= 0;
        end else if (slv_reg_wren) begin
            case (axi_awaddr[ADDR_LSB + NUM_REG_BITS - 1 : ADDR_LSB])
                'h0: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                        if (S_AXI_WSTRB[byte_index]) slv_reg0[(byte_index*8)+:8] <= S_AXI_WDATA[(byte_index*8)+:8];
                'h2: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                        if (S_AXI_WSTRB[byte_index]) slv_reg2[(byte_index*8)+:8] <= S_AXI_WDATA[(byte_index*8)+:8];
                'h3: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                        if (S_AXI_WSTRB[byte_index]) slv_reg3[(byte_index*8)+:8] <= S_AXI_WDATA[(byte_index*8)+:8];
                'h4: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                        if (S_AXI_WSTRB[byte_index]) slv_reg4[(byte_index*8)+:8] <= S_AXI_WDATA[(byte_index*8)+:8];
                'h5: for (byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1)
                        if (S_AXI_WSTRB[byte_index]) slv_reg5[(byte_index*8)+:8] <= S_AXI_WDATA[(byte_index*8)+:8];
                default: ;
            endcase
        end
    end

    // =========================================================================
    // AXI Write Response Channel
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_bvalid <= 0;
            axi_bresp  <= 2'b0;
        end else if (axi_awready && S_AXI_AWVALID && ~axi_bvalid && axi_wready && S_AXI_WVALID) begin
            axi_bvalid <= 1'b1;    // Write response valid
            axi_bresp  <= 2'b0;    // OKAY response
        end else if (S_AXI_BREADY && axi_bvalid) begin
            axi_bvalid <= 1'b0;
        end
    end

    // =========================================================================
    // AXI Read Address Channel FSM
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 0;
        end else if (~axi_arready && S_AXI_ARVALID) begin
            axi_arready <= 1'b1;           // Accept read address
            axi_araddr  <= S_AXI_ARADDR;   // Latch read address
        end else begin
            axi_arready <= 1'b0;
        end
    end

    // =========================================================================
    // AXI Read Data Channel FSM
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rvalid <= 0;
            axi_rresp  <= 0;
        end else if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
            axi_rvalid <= 1'b1;
            axi_rresp  <= 2'b0;            // OKAY response
        end else if (axi_rvalid && S_AXI_RREADY) begin
            axi_rvalid <= 1'b0;
        end
    end

    // =========================================================================
    // Read Data Output Mux
    // =========================================================================
    always @(*) begin
        case (axi_araddr[ADDR_LSB + NUM_REG_BITS - 1 : ADDR_LSB])
            'h0: reg_data_out = slv_reg0; // Control
            'h1: reg_data_out = slv_reg1; // Status
            'h2: reg_data_out = slv_reg2; // A00_A01
            'h3: reg_data_out = slv_reg3; // A10_A11
            'h4: reg_data_out = slv_reg4; // B00_B01
            'h5: reg_data_out = slv_reg5; // B10_B11
            'h6: reg_data_out = slv_reg6; // C00
            'h7: reg_data_out = slv_reg7; // C01
            'h8: reg_data_out = slv_reg8; // C10
            'h9: reg_data_out = slv_reg9; // C11
            'hA: reg_data_out = debug_reg; // Debug
            default: reg_data_out = 0;
        endcase
    end

    // Output read data register
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            axi_rdata <= 0;
        else if (axi_arready && S_AXI_ARVALID && ~axi_rvalid)
            axi_rdata <= reg_data_out;
    end

    // =========================================================================
    // NPU FSM and Arithmetic Logic
    // =========================================================================

    // Operand and result widths
    localparam DATA_WIDTH = 8;         // Matrix operand width
    localparam RESULT_WIDTH = 17;      // Product width (signed)

    // FSM state definitions
    localparam S_IDLE     = 3'b000;
    localparam S_LOAD     = 3'b001;
    localparam S_COMPUTE  = 3'b010;
    localparam S_ACTIVATE = 3'b011;
    localparam S_STORE    = 3'b100;
    localparam S_DONE     = 3'b101;
    reg [2:0] npu_state;

    // Matrix registers: signed for correct arithmetic
    reg signed [DATA_WIDTH-1:0]   a00, a01, a10, a11;
    reg signed [DATA_WIDTH-1:0]   b00, b01, b10, b11;
    reg signed [RESULT_WIDTH-1:0] c00, c01, c10, c11;
    reg signed [RESULT_WIDTH-1:0] c00_act, c01_act, c10_act, c11_act;

    reg npu_done, npu_busy;

    // Edge detector for start bit in control register (slv_reg0[0])
    reg prev_start;
    wire start_rising;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            prev_start <= 1'b0;
        else
            prev_start <= slv_reg0[0];
    end
    assign start_rising = slv_reg0[0] & ~prev_start;

    // =========================================================================
    // Main NPU FSM: Matrix Load, Compute, Activate (ReLU), Store, Done
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            npu_state <= S_IDLE;
            npu_busy  <= 1'b0;
            npu_done  <= 1'b0;
            a00 <= 0; a01 <= 0; a10 <= 0; a11 <= 0;
            b00 <= 0; b01 <= 0; b10 <= 0; b11 <= 0;
            c00 <= 0; c01 <= 0; c10 <= 0; c11 <= 0;
            c00_act <= 0; c01_act <= 0; c10_act <= 0; c11_act <= 0;
            debug_reg <= 32'hDEADBEEF;
        end else begin
            // Clear "done" flag on request
            if (slv_reg0[1]) begin
                npu_done <= 1'b0;
            end

            case (npu_state)
                S_IDLE: begin
                    npu_busy <= 1'b0;
                    if (start_rising) begin
                        npu_busy  <= 1'b1;
                        npu_done  <= 1'b0;
                        npu_state <= S_LOAD;
                        debug_reg <= 32'hAAAAAAAA; // IDLE->LOAD marker
                    end else begin
                        debug_reg <= 32'h55555555; // Still IDLE
                    end
                end
                S_LOAD: begin
                    // Unpack inputs from AXI slave registers
                    a00 <= slv_reg2[DATA_WIDTH-1:0];   a01 <= slv_reg2[DATA_WIDTH*2-1:DATA_WIDTH];
                    a10 <= slv_reg3[DATA_WIDTH-1:0];   a11 <= slv_reg3[DATA_WIDTH*2-1:DATA_WIDTH];
                    b00 <= slv_reg4[DATA_WIDTH-1:0];   b01 <= slv_reg4[DATA_WIDTH*2-1:DATA_WIDTH];
                    b10 <= slv_reg5[DATA_WIDTH-1:0];   b11 <= slv_reg5[DATA_WIDTH*2-1:DATA_WIDTH];
                    debug_reg <= 32'hBBBBBBBB; // LOAD->COMPUTE
                    npu_state <= S_COMPUTE;
                end
                S_COMPUTE: begin
                    // 2x2 elementwise multiply (dot product)
                    c00 <= $signed(a00) * $signed(b00);
                    c01 <= $signed(a01) * $signed(b01);
                    c10 <= $signed(a10) * $signed(b10);
                    c11 <= $signed(a11) * $signed(b11);
                    npu_state <= S_ACTIVATE;
                end
                S_ACTIVATE: begin
                    // Optional: apply ReLU to each result if enabled
                    c00_act <= (slv_reg0[8] && c00[RESULT_WIDTH-1]) ? 0 : c00;
                    c01_act <= (slv_reg0[8] && c01[RESULT_WIDTH-1]) ? 0 : c01;
                    c10_act <= (slv_reg0[8] && c10[RESULT_WIDTH-1]) ? 0 : c10;
                    c11_act <= (slv_reg0[8] && c11[RESULT_WIDTH-1]) ? 0 : c11;
                    debug_reg <= 32'hACACACAC; // ACTIVATE->STORE
                    npu_state <= S_STORE;
                end
                S_STORE: begin
                    // Output results (exposed via output slave registers)
                    debug_reg <= 32'hDDDDDDDD; // STORE->DONE
                    npu_state <= S_DONE;
                end
                S_DONE: begin
                    npu_done  <= 1'b1;   // Signal done
                    npu_busy  <= 1'b0;
                    debug_reg <= 32'hEEEEEEEE; // DONE->IDLE
                    npu_state <= S_IDLE;
                end
                default: npu_state <= S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Output Register Update (status, result, etc.)
    // =========================================================================
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            slv_reg1 <= 0;
            slv_reg6 <= 0; slv_reg7 <= 0; slv_reg8 <= 0; slv_reg9 <= 0;
        end else begin
            // slv_reg1[0]: busy, slv_reg1[1]: done
            slv_reg1 <= {{(C_S_AXI_DATA_WIDTH-2){1'b0}}, npu_done, npu_busy};

            // Write outputs in S_STORE state
            if (npu_state == S_STORE) begin
                slv_reg6 <= c00_act;
                slv_reg7 <= c01_act;
                slv_reg8 <= c10_act;
                slv_reg9 <= c11_act;
            end
        end
    end

endmodule
